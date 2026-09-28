"""Checks training.grad_mode='first_order' against the default create_graph path on one batch.

Reports the cosine similarity / relative error of the parameter gradients and, on CUDA, the peak memory
and time per training step of both modes. Uses a random batch in [-1, 1] (no dataset download needed).

  python3 check_first_order.py --config ./configs/homotopy/cifar10.py --batch_size 128

The finite difference in 'first_order' needs full fp32 arithmetic; TF32 is disabled below. Run once with
--allow_tf32 to see the effect on your hardware before enabling TF32 for 'first_order' training.
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
  parser.add_argument('--fd_step', type=float, default=None)
  parser.add_argument('--iters', type=int, default=3)
  parser.add_argument('--allow_tf32', action='store_true')
  args = parser.parse_args()

  torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
  torch.backends.cudnn.allow_tf32 = args.allow_tf32

  config = load_config(args.config)
  if args.batch_size is not None:
    config.training.batch_size = config.training.small_batch_size = args.batch_size
  if args.fd_step is not None:
    config.training.fd_step = args.fd_step
  device = config.device
  sde = methods.Homotopy(config)
  torch.manual_seed(0)
  model = mutils.create_model(config)
  loss_fn = losses.get_perturb_batch_loss_fn(sde, train=True, method_name='homotopy')
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

  grads, values = {}, {}
  for mode in ('double', 'first_order'):
    values[mode] = step(mode, seed=1234)
    grads[mode] = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
  ref, new = grads['double'], grads['first_order']
  cos = torch.nn.functional.cosine_similarity(new, ref, dim=0).item()
  rel = ((new - ref).norm() / ref.norm()).item()
  print(f"fd_step={config.training.fd_step:g}  TF32={args.allow_tf32}")
  print(f"loss: double={values['double']:.6f}  first_order={values['first_order']:.6f}")
  print(f"parameter gradient: cosine={cos:.6f}  relative L2 error={rel:.2e}")

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
