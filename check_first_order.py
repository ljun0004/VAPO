"""Checks training.grad_mode='first_order' against the default create_graph path on one batch.

Reports the cosine similarity / relative error of the parameter gradients and, on CUDA, the peak memory
and time per training step of both modes.

  python3 check_first_order.py --config ./configs/homotopy/cifar10.py --batch_size 128

The finite-difference error depends on the curvature of the network, so calibrate fd_step on a trained model
and real data, e.g.

  python3 check_first_order.py --checkpoint homotopy_cifar10/checkpoints/checkpoint_30.pth --ema \
      --batch_npy batch.npy --fd_step 0.01,0.03,0.1,0.3

where batch.npy holds a [B, C, H, W] array already scaled like the training data ([-1, 1] when
data.centered). Without --batch_npy a random batch in [-1, 1] is used.

The finite difference in 'first_order' needs full fp32 arithmetic, so it disables TF32 in its own passes
unless training.fd_allow_tf32 is set (--allow_tf32). The gradient reference ('double') is computed in strict
fp32; the timings of 'double' use PyTorch's default TF32 flags, i.e. what training uses today.
"""

import argparse
import importlib.util
import sys
import time
import types

import torch

try:
  import datasets  # noqa: F401  (imported by losses.py; pulls in tensorflow)
except ImportError:
  sys.modules['datasets'] = types.ModuleType('datasets')

import losses
import methods
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import models.unet  # noqa: F401  (registers the 'unet' model)


def load_config(path):
  spec = importlib.util.spec_from_file_location('vapo_config', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.get_config()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default='./configs/homotopy/cifar10.py')
  parser.add_argument('--batch_size', type=int, default=None)
  parser.add_argument('--fd_step', default=None, help='finite-difference step, or a comma-separated list to sweep')
  parser.add_argument('--checkpoint', default=None, help='checkpoint .pth saved by run_lib.train')
  parser.add_argument('--ema', action='store_true', help='evaluate the EMA weights of --checkpoint')
  parser.add_argument('--batch_npy', default=None, help='.npy array [B, C, H, W] of scaled training images')
  parser.add_argument('--iters', type=int, default=3)
  parser.add_argument('--allow_tf32', action='store_true')
  args = parser.parse_args()

  config = load_config(args.config)
  if args.batch_size is not None:
    config.training.batch_size = config.training.small_batch_size = args.batch_size
  fd_steps = [float(h) for h in args.fd_step.split(',')] if args.fd_step else [config.training.fd_step]
  config.training.fd_allow_tf32 = args.allow_tf32
  device = config.device
  sde = methods.Homotopy(config)
  torch.manual_seed(0)
  model = mutils.create_model(config)
  if args.checkpoint:
    loaded = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(loaded['model'], strict=False)
    if args.ema:
      ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
      ema.load_state_dict(loaded['ema'])
      ema.copy_to(model.parameters())
  loss_fn = losses.get_perturb_batch_loss_fn(sde, train=True, method_name='homotopy')
  if args.batch_npy:
    import numpy as np
    batch = torch.from_numpy(np.load(args.batch_npy)).float().to(device)
    config.training.batch_size = config.training.small_batch_size = batch.shape[0]
  else:
    B, C, H = config.training.batch_size, config.data.channels, config.data.image_size
    batch = (torch.rand(B, C, H, H) * 2 - 1).to(device)
  params = [p for p in model.parameters() if p.requires_grad]

  def step(mode, seed):
    config.training.grad_mode = mode
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    loss = loss_fn(model, batch, {'step': 0}, True)[0]
    if loss.requires_grad:
      loss.backward()
    return float(loss)

  default_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
  print(f"fd_allow_tf32={args.allow_tf32}  default TF32 flags (matmul, cudnn) used for 'double' timing: {default_tf32}")

  def grad_of(mode):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # strict fp32 reference
    value = step(mode, seed=1234)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = default_tf32
    return value, torch.cat([p.grad.flatten() for p in params if p.grad is not None])

  value_ref, ref = grad_of('double')
  for h in fd_steps:
    config.training.fd_step = h
    value, new = grad_of('first_order')
    cos = torch.nn.functional.cosine_similarity(new, ref, dim=0).item()
    rel = ((new - ref).norm() / ref.norm()).item()
    print(f"fd_step={h:g}: gradient cosine={cos:.6f}  relative L2 error={rel:.2e}  "
          f"(loss double={value_ref:.6f}, first_order={value:.6f})")
  config.training.fd_step = fd_steps[0]

  for mode in ('double', 'first_order'):
    if device.type == 'cuda':
      torch.cuda.synchronize()
      torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(args.iters):
      t0 = time.perf_counter()
      step(mode, seed=i)
      if device.type == 'cuda':
        torch.cuda.synchronize()
      times.append(time.perf_counter() - t0)
    mem = f"  peak memory={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if device.type == 'cuda' else ''
    print(f"{mode:>12}: {min(times):.3f} s/step{mem}")


if __name__ == '__main__':
  main()
