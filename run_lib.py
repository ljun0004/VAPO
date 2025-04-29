# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for PFGM or score-based generative models. """

import gc
import io
import os
import time
import copy
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import unet
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import methods
from absl import flags
import torch
torch.cuda.empty_cache()
from torch.utils import tensorboard
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint, ContinueIteration
from models.utils import get_predict_fn, from_flattened_numpy
import datasets_utils.celeba
from info_gpu import _get_gpu_usage
from mask import get_masked_fn


FLAGS = flags.FLAGS
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  net = mutils.create_model(config)
  ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, net.parameters())
  state = dict(optimizer=optimizer, model=net, ema=ema, step=0, sigma_max=config.training.sigma_max, t_eval=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  logging.info("Weight decay: %.5f" %state['optimizer'].param_groups[0]['weight_decay'])
  initial_step = int(state['step'])

  # Build data iterators
  if config.data.dataset == 'CELEBA':
    # I cannot load CelebA from tfds loader. So I write a pytorch loader instead.
    train_ds, eval_ds = datasets_utils.celeba.get_celeba(config)
  else:
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  # Setup methods
  if config.training.sde.lower() == 'vpsde':
    sde = methods.VPSDE(config=config, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = methods.subVPSDE(config=config, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = methods.VESDE(config=config, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'poisson':
    # PFGM
    sde = methods.Poisson(config=config)
    sampling_eps = config.sampling.z_min
  elif config.training.sde.lower() == 'homotopy':
    # GrAPH
    sde = methods.Homotopy(config=config)
    sampling_eps = config.sampling.eps_z; rtol = config.sampling.rtol; atol = config.sampling.atol
  else:
    raise NotImplementedError(f"Method {config.training.sde} unknown.")

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (100, config.data.channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, rtol, atol)

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  reduce_mean = config.training.reduce_mean
  method_name = config.training.sde.lower()
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn, sampling_fn=sampling_fn,
                                     reduce_mean=reduce_mean, method_name=method_name)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, method_name=method_name)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info(torch.__version__)
  logging.info(torch.cuda.is_available())
  logging.info(workdir)
  logging.info("batch_size: %d, small_batch_size: %d, num_particles: %d, sample_size: %d, sample_freq: %d, reduce_mean: %s, scale_by_sigma: %s, action_matching: %s, entropic: %s, coeff_diffusion: %f" 
               % (config.training.batch_size, config.training.small_batch_size, config.training.num_particles, config.training.sample_size, config.training.sample_freq, config.training.reduce_mean, config.model.scale_by_sigma, config.training.action_matching, config.training.entropic, config.training.coeff_diffusion))
  logging.info("data_scale: %f, data_centered: %s, mean_prior: %f, sigma_dec_min: %f, sigma_dec_max: %f, sigma_clip: %f, sigma_anneal: %f" 
               % (config.data.scale, config.data.centered, config.training.mean_prior, config.training.sigma_min, config.training.sigma_max, config.training.sigma_clip, 1 - config.training.sigma_anneal))
  logging.info("std_power_cov: %f, std_power_vel: %f, std_power_lap: %f, std_power_norm: %f, std_power_temb: %f"
               % (config.training.std_power_cov, config.training.std_power_vel, config.training.std_power_lap, config.training.std_power_norm, config.training.std_power_temb))
  logging.info("weight_cov: %f, weight_vel: %f, weight_lap: %f, weight_norm: %f, weight_temb: %f" 
               % (config.training.weight_cov, config.training.weight_vel, config.training.weight_lap, config.training.weight_norm, config.training.weight_temb))
  logging.info("divisor: %f, sigma_prior: %f, eps_sigma: %f, invert_sigma: %s eps_z: %e, augment_z: %s, augment_t: %s, class_guidance: %s, p_uncond: %s" 
               % (config.training.divisor, config.training.sigma_prior, config.training.eps_sigma, config.training.invert_sigma, config.sampling.eps_z, config.training.augment_z, config.training.augment_t, config.training.class_guidance, config.training.p_uncond))
  logging.info("t_start: %d, t_max: %d, t_end: %d, scheduling: %s, shift: %s, mean_power: %f, var_power: %f, std_power_cov: %f, std_power_norm: %f, eps_max: %e, eps_min: %e, invert_t: %s, snapshot_freq: %d, checkpoint_freq: %d" 
               % (config.training.t_start, config.training.t_max, config.training.t_end, config.training.scheduling, config.training.shift_schedule, config.training.mean_power, config.training.var_power, config.training.std_power_cov, config.training.std_power_norm, config.training.eps_max, config.training.eps_min, config.training.invert_t, config.training.snapshot_freq, config.training.snapshot_freq_for_preemption))
  logging.info("learning_rate: %e, weight_decay: %e, alpha: %f, delta: %f, data_init: %s" 
               % (config.optim.lr, config.optim.weight_decay, config.optim.alpha, config.optim.delta, config.model.data_init))
  logging.info("optimizer: %s, beta1: %f, grad_clip: %d, warmup: %d, anneal_rate: %f, anneal_iters: %s, cov: %s, norm: %s, laplacian: %s, laplacian_mode: %s, cvf_mode: %s" 
               % (config.optim.optimizer, config.optim.beta1, config.optim.grad_clip, config.optim.warmup, config.optim.anneal_rate, config.optim.anneal_iters, config.optim.cov, config.optim.norm, config.optim.laplacian, config.optim.laplacian_mode, config.optim.cvf_mode))
  logging.info("num_channels: %d, num_res_blocks: %d, num_heads: %d, multiplier: %f, divisor: %f" 
               % (config.model.nf, config.model.num_res_blocks, config.model.num_heads, config.model.multiplier, config.model.divisor))
  logging.info("%s, num_params: %e" %(config.model.name, sum(p.numel() for p in net.parameters())))
  logging.info("Starting training loop at step %d." % (initial_step,))
  
  for step in range(initial_step, num_train_steps + 1):
    # # Generate and save samples
    # if config.training.snapshot_sampling and step == initial_step and step != 0:
    #   that_sample_dir = os.path.join(sample_dir, "iter_{}".format(step-1))
    #   if not os.path.exists(that_sample_dir):
    #     ema.store(net.parameters())
    #     ema.copy_to(net.parameters())
    #     sample, n = sampling_fn(net, state)
    #     logging.info("step: %d, nfe_sampling: %d" % (step, n))
    #     ema.restore(net.parameters())
    #     this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
    #     tf.io.gfile.makedirs(this_sample_dir)
    #     nrow = int(np.sqrt(sample.shape[0]))
    #     image_grid = make_grid(sample, nrow, padding=2)
    #     sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    #     with tf.io.gfile.GFile(
    #         os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
    #       np.save(fout, sample)

    #     with tf.io.gfile.GFile(
    #         os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
    #       save_image(image_grid, fout)

    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    if config.data.dataset == 'CELEBA':
      try:
        batch, labels = next(train_iter)
        batch = batch.cuda()
        labels = labels.cuda()
        if len(batch) != config.training.batch_size:
          continue
      except StopIteration:
        train_iter = iter(train_ds)
        batch, labels = next(train_iter)
        batch = batch.cuda()
        labels = labels.cuda()
    else:
      batch_ds = next(train_iter)
      batch = torch.from_numpy(batch_ds['image']._numpy()).to(config.device).float()
      batch = batch.permute(0, 3, 1, 2)
      # JULIA's CEMB change Get labels and do one-hot encoding
      labels = torch.from_numpy(batch_ds['label']._numpy()).to(config.device).long()
      # labels = None
    batch = scaler(batch)

    # Execute one training step
    # Julia's changes: add labels here
    loss, corr, lap, norm, reg = train_step_fn(state, batch, labels=labels)
    if step == initial_step: logging.info(_get_gpu_usage())
    if step % config.training.log_freq == 0:
      logging.info("step: %d, loss: %.5e, cov: %.5e, lap: %.5e, norm: %.5e, reg: %.5e" % (step, loss.item(), corr.item(), lap.item(), norm.item(), reg.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      print('saving checkpoint...')
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      if config.data.dataset == 'CELEBA':
        try:
          eval_batch, eval_labels = next(eval_iter)
          eval_batch = eval_batch.cuda()
          eval_labels = eval_labels.cuda()
          if len(eval_batch) != config.training.batch_size:
            continue
        except StopIteration:
          eval_iter = iter(eval_ds)
          eval_batch, eval_labels = next(eval_iter)
          eval_batch = eval_batch.cuda()
          eval_labels = eval_labels.cuda()
      else:
        eval_batch_ds = next(eval_iter)
        eval_batch = torch.from_numpy(eval_batch_ds['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        # JULIA's CEMB change Get labels and do one-hot encoding
        eval_labels = torch.from_numpy(eval_batch_ds['label']._numpy()).to(config.device).long()
        # labels = None
      eval_batch = scaler(eval_batch)
      eval_loss, eval_corr, eval_lap, eval_norm, eval_reg = eval_step_fn(state, eval_batch, labels=eval_labels)
      logging.info("step: %d, eval_loss: %.5e, eval_cov: %.5e, eval_lap: %.5e, eval_norm: %.5e, eval_reg: %.5e" % (step, eval_loss.item(), eval_corr.item(), eval_lap.item(), eval_norm.item(), eval_reg.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    # print(config.training.snapshot_freq, config.training.snapshot_freq_for_preemption)
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step
      print('sampling snapshot...')
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(net.parameters())
        ema.copy_to(net.parameters())
        sample, n = sampling_fn(net, state)
        logging.info("step: %d, nfe_sampling: %d" % (step, n))
        ema.restore(net.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)


def evaluate(config,
             workdir,
             eval_folder="eval",
             save_folfer="save"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder

  # set random seed
  tf.random.set_seed(config.seed)

  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  save_dir = os.path.join(workdir, save_folfer)
  tf.io.gfile.makedirs(save_dir)

  # Build data pipeline

  if not config.eval.save_images:
    if config.data.dataset == 'CELEBA':
      train_ds, eval_ds = datasets_utils.celeba.get_celeba(config)
    else:
      train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                  uniform_dequantization=config.data.uniform_dequantization,
                                                  evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  net = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, net.parameters())
  ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=net, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  logging.info("%s, num_params: %e" %(config.model.name, sum(p.numel() for p in net.parameters())))

  # Setup methods
  if config.training.sde.lower() == 'vpsde':
    sde = methods.VPSDE(config=config, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.sampling.N)
    sampling_eps = 1e-3; rtol=1e-4; atol=1e-4
  elif config.training.sde.lower() == 'subvpsde':
    sde = methods.subVPSDE(config=config, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3; rtol=1e-4; atol=1e-4
  elif config.training.sde.lower() == 'vesde':
    sde = methods.VESDE(config=config, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5; rtol=1e-4; atol=1e-4
  elif config.training.sde.lower() == 'poisson':
    # PFGM
    sde = methods.Poisson(config=config)
    sampling_eps = config.sampling.z_min; rtol=1e-4; atol=1e-4
  elif config.training.sde.lower() == 'homotopy':
    # GrAPH
    sde = methods.Homotopy(config=config)
    sampling_eps = config.sampling.eps_z; rtol = config.sampling.rtol; atol = config.sampling.atol
  else:
    raise NotImplementedError(f"Method {config.training.sde} unknown.")
  print("--- sampling eps:", sampling_eps)

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   method_name=config.training.sde.lower())


  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
      ds_bpd = train_ds_bpd
      bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
      # Go over the dataset 5 times when computing likelihood on the test dataset
      ds_bpd = eval_ds_bpd
      bpd_num_repeats = 5
    else:
      raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")
    if config.training.sde.lower() in ['homotopy','poisson']:
      likelihood_fn = likelihood.get_likelihood_fn_pfgm(sde)
    else:
      likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, rtol, atol)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  calculate_inception_score = config.eval.calc_is
  print(f"inceptionv3: {inceptionv3}, calc IS: {calculate_inception_score}")
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))

  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1, config.eval.eval_increment):

    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(config.seed)

    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt * config.training.snapshot_freq))
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt * config.training.snapshot_freq}.pth')

    if not tf.io.gfile.exists(ckpt_filename):
      print(f"{ckpt_filename} does not exist")
      continue

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    print("loading from ", ckpt_path)
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(net.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      print("please don't set the config.eval.save_images flag, or the datasets wouldn't be loaded.")
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss, _, _, _, _ = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(net, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    if config.eval.enable_interpolate:

      from scipy.spatial import geometric_slerp
      repeat = 6
      inter_num = 10

      sampling_shape = (inter_num,
                        config.data.channels,
                        config.data.image_size, config.data.image_size)
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, rtol, atol)
      imgs = torch.empty(
        (repeat * inter_num, config.data.channels, config.data.image_size, config.data.image_size))

      for i in range(repeat):
        N = np.prod(sampling_shape[1:])
        gaussian = torch.randn(2, N).cuda()
        unit_vec = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        t_vals = np.linspace(0, 1, inter_num)
        # spherical interpolations
        unit_vec = unit_vec.detach().cpu().numpy().astype(np.double)
        unit_vec /= np.sqrt(np.sum(unit_vec ** 2, axis=1, keepdims=True))
        result = geometric_slerp(unit_vec[0], unit_vec[1], t_vals)
        result = result * config.sampling.upper_norm
        result = torch.from_numpy(result).cuda()

        samples, n = sampling_fn(net, x=result)
        imgs[i * inter_num: (i + 1) * inter_num] = torch.clamp(samples, 0.0, 1.0).to('cpu')

      image_grid = make_grid(imgs, nrow=inter_num)
      save_image(image_grid, os.path.join(eval_dir, f'interpolation_{ckpt}.png'))

    if config.eval.enable_rescale:

      from scipy.spatial import geometric_slerp
      repeat = 6
      inter_num = 10

      sampling_shape = (inter_num,
                        config.data.channels,
                        config.data.image_size, config.data.image_size)
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, rtol, atol)
      imgs = torch.empty(
        (repeat * inter_num, config.data.channels, config.data.image_size, config.data.image_size))

      for i in range(repeat):
        N = np.prod(sampling_shape[1:])
        gaussian = torch.randn(1, N).cuda()
        unit_vec = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        t_vals = torch.linspace(1000, 6000, inter_num).cuda()
        t_vals = t_vals.view((-1, 1))
        result = unit_vec * t_vals

        samples, n = sampling_fn(net, x=result)
        imgs[i * inter_num: (i + 1) * inter_num] = torch.clamp(samples, 0.0, 1.0).to('cpu')

      image_grid = make_grid(imgs, nrow=inter_num)
      save_image(image_grid, os.path.join(eval_dir, f'rescale_{ckpt}.png'))

    def load_batch(ds, ds_iter, is_ood=False):
      if config.data.dataset == 'CELEBA' or (is_ood and config.eval.classify_dataset.data.dataset == 'CELEBA'):
        # print("trigerred celeba load_batch")
        try:
          batch = next(ds_iter)[0].cuda()
          batch_size = config.eval.classify_dataset.eval.batch_size if is_ood else config.eval.batch_size
          if len(batch) != batch_size:
            raise ContinueIteration
        except StopIteration:
          ds_iter = iter(ds)
          batch = next(ds_iter)[0].cuda()
      else:
        batch = torch.from_numpy(next(ds_iter)['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
      batch = scaler(batch)
      return batch, ds_iter

    def get_model_out(samples_x, boltzmann=False):
      # samples_x: [B, C, H, W]
      samples_x = samples_x.flatten(1, 3).to(config.device).type(torch.float32) # [B, C*H*W]
      # print(samples_x.shape)
      # Predict scalar potential (FC)
      if sde.config.training.augment_t:
          # [B, 1]
          t_samples = torch.ones(len(samples_x), 1).to(config.device).type(torch.float32)
          std_enc = torch.zeros(len(samples_x), 1).to(config.device).type(torch.float32)
          samples_net = torch.cat([samples_x, t_samples, std_enc], dim=-1)
      else:
          samples_net = samples_x
      # print(samples_net.shape)
      psi = net_fn(samples_net).squeeze(dim=-1)
      if boltzmann:
        f = -1; g = 1
        psi = (4 * psi + 1 * f * samples_x.pow(2).sum(dim=-1)) / (g**2)
        # psi = psi - psi.mean()
        # psi = psi / (psi.std() + 1e-6)
        psi = torch.log(psi)
      return psi
    
    # store model outputs for plotting histogram
    if config.eval.get_model_outs:
      print("GETTING MODEL OUT")
      # Get model function
      net_fn = get_predict_fn(sde, net, train=False)

      train_outs_all = None
      eval_outs_all = None

      # NOTE: below doesn't work for CELEBA
      print(f"length of train ds: {train_ds.cardinality()}\nlength of eval ds: {eval_ds.cardinality()}")

      for i, (ds, n_iters) in enumerate([(train_ds, train_ds.cardinality()), 
                                         (eval_ds, eval_ds.cardinality())]):
        ds_iter = iter(ds)
        for step in tqdm(range(n_iters)): # iterate through dataset
          try:
            batch, ds_iter = load_batch(ds, ds_iter) # batch: [B, C, H, W]
          except ContinueIteration:
            continue

          # get model output
          out = get_model_out(batch) # Tensor[B]
          # print(out)

          with torch.no_grad():
            if i == 0: # train
              train_outs_all = out if train_outs_all is None else torch.cat([train_outs_all, out], 0)
            else:
              eval_outs_all = out if eval_outs_all is None else torch.cat([eval_outs_all, out], 0)

      if config.eval.save_model_outs:
        with open("train_outs_all.pkl", "wb") as f:
          pickle.dump(train_outs_all, f)

        with open("eval_outs_all.pkl", "wb") as f:
          pickle.dump(eval_outs_all, f)

      print("done getting model outs. exiting...")
      return
    
    if config.eval.classify_data_distribution:
      print("CLASSIFYING DATA DIST AS ID OR OOD")

      if config.eval.classify_dataset.data.dataset == 'CELEBA':
        print("OOD IS CELEBA")
        # I cannot load CelebA from tfds loader. So I write a pytorch loader instead.
        _, cls_eval_ds = datasets_utils.celeba.get_celeba(config.eval.classify_dataset)
        cls_eval_ds_len = len(cls_eval_ds)
      else:
        _, cls_eval_ds, _ = datasets.get_dataset(config.eval.classify_dataset,
                                                  uniform_dequantization=config.eval.classify_dataset.data.uniform_dequantization,
                                                  evaluation=True)
        cls_eval_ds_len = cls_eval_ds.cardinality()
      # Get model function
      net_fn = get_predict_fn(sde, net, train=False)
      # NOTE: below doesn't work for CELEBA
      print(f"length of OOD eval ds: {cls_eval_ds_len}")

      num_ood = 0
      num_id = 0
      
      # get model outs for ID
      eval_outs_all_id = None

      eval_ds_iter = iter(eval_ds)
      for step in tqdm(range(eval_ds.cardinality())): # iterate through dataset
        try:
          batch, eval_ds_iter = load_batch(eval_ds, eval_ds_iter) # batch: [B, C, H, W]
        except ContinueIteration:
          continue

        # get model output
        out = get_model_out(batch, boltzmann=True) # Tensor[B]

        with torch.no_grad():
          eval_outs_all_id = out if eval_outs_all_id is None else torch.cat([eval_outs_all_id, out], 0)

      ## get model outs for OOD
      eval_outs_all_ood = None

      for i, (ds, n_iters) in enumerate([(cls_eval_ds, cls_eval_ds_len)]):
        ds_iter = iter(ds)
        for step in tqdm(range(n_iters)): # iterate through dataset
          try:
            batch, ds_iter = load_batch(ds, ds_iter, is_ood=True) # batch: [B, C, H, W]
          except ContinueIteration:
            continue

          if step < 5: print(f"batch size: {batch.shape}")
          # get model output
          out = get_model_out(batch, boltzmann=True) # Tensor[B]

          with torch.no_grad():
            # classify ood
            # print(f"lbound: {config.eval.ood_lbound}, ubound: {config.eval.ood_ubound}")
            # print(out)
            # ood = torch.logical_or(out < config.eval.ood_lbound,
            #                        out > config.eval.ood_ubound)
            # num_ood += ood.sum()
            # num_id += (~ood).sum()

            eval_outs_all_ood = out if eval_outs_all_ood is None else torch.cat([eval_outs_all_ood, out], 0)
        
        # reset for evaluation ds
        # num_ood = 0
        # num_id = 0
      with torch.no_grad():
        # eval_outs_all_id = (eval_outs_all_id - eval_outs_all_id.min()) / (eval_outs_all_id.max() - eval_outs_all_id.min())
        # eval_outs_all_ood = (eval_outs_all_ood - eval_outs_all_ood.min()) / (eval_outs_all_ood.max() - eval_outs_all_ood.min())
        
        scores = torch.cat([eval_outs_all_id, eval_outs_all_ood]).detach().cpu().numpy()
        # scores = ((scores - scores.min()) / (scores.max() - scores.min())).detach().cpu().numpy()
        print(f"scores.shape: {scores.shape}")
        labels = torch.cat([torch.ones_like(eval_outs_all_id), torch.zeros_like(eval_outs_all_ood)]).detach().cpu().numpy()
        print(f"labels.shape: {labels.shape}")

        score = roc_auc_score(labels, scores)

      print(f"for OOD dataset: {config.eval.classify_dataset.data.dataset}, roc auc score: {score}")

      if config.eval.save_model_outs:
        with open(f"train_outs_all_{config.eval.classify_dataset.data.dataset}.pkl", "wb") as f:
          pickle.dump(train_outs_all, f)

        with open(f"eval_outs_all_{config.eval.classify_dataset.data.dataset}.pkl", "wb") as f:
          pickle.dump(eval_outs_all, f)

      print("done classifying data distribution. Exiting...")
      return

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = (config.eval.num_samples // config.eval.batch_size) + 1
      logging.info("number of sampling rounds %d" % num_sampling_rounds)
      # Directory to save samples. Different for each host to avoid writing conflicts
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      tf.io.gfile.makedirs(this_sample_dir)
      this_save_dir = os.path.join(save_dir, f"ckpt_{ckpt}")
      tf.io.gfile.makedirs(this_save_dir)

      def single_sampling_iter(iter_id, x=None, mask=None, grad_mask=None, original_x=None):
        print("sampling iter")
        logging.info("sampling -- ckpt: %d, round: %d/%d" % (ckpt, iter_id + 1, num_sampling_rounds))

        # ema.store(net.parameters())
        # ema.copy_to(net.parameters())
        samples, n = sampling_fn(net, state, x=x, mask=mask, grad_mask=grad_mask)
        print("nfe:", n, "shape:", samples.shape)
        # ema.restore(net.parameters())
        samples_torch = copy.deepcopy(samples)
        samples_torch = samples_torch.view(-1, config.data.channels, config.data.image_size, config.data.image_size)

        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.channels))

        if config.eval.save_samples:
          # Write samples to disk or Google Cloud Storage
          print("saving samples to:", os.path.join(this_sample_dir, f"samples_{iter_id}.npz"))
          # print(original_x)
          if original_x is not None:
            with tf.io.gfile.GFile(
                  os.path.join(this_sample_dir, f"original_{iter_id}.npz"), "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, samples=original_x.detach().cpu())
              fout.write(io_buffer.getvalue())
          with tf.io.gfile.GFile(
                  os.path.join(this_sample_dir, f"samples_{iter_id}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())

        if config.eval.save_images:
          # Saving a few generated images for debugging / visualization
          # image_grid = make_grid(samples_torch, nrow=int(np.sqrt(len(samples_torch))))
          # save_image(image_grid, os.path.join(eval_dir, f'ode_images_{ckpt}.png'))
          # exit(0)
          print("saving images to:", this_save_dir)
          for i, sample in enumerate(samples_torch):
            save_image(sample, os.path.join(this_save_dir, f'image_{iter_id}_{i}.png'))

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                      inceptionv3=inceptionv3,
                                                      ret_logits=calculate_inception_score)
        # print(f"inceptionv3: {inceptionv3}")
        # print(f'latents["logits"]: {latents["logits"]}')
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{iter_id}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

        # return samples, n
      
      # for image inpainting
      if config.eval.mask:
        # NOTE: assumes CelebA (if not, then need to change dataset pre-processing)
        print("USING MASK")
        with torch.no_grad():
          mask_fn = get_masked_fn(config)
          
          # NOTE: assume mask is fixed for all images, if not move this to within the loop below
          # 
          # with torch.no_grad():
          mask_init = ~mask_fn(config).bool() # [H, W]
          eval_n_iters = len(eval_ds)
          eval_ds_iter = iter(eval_ds)
          for i in tqdm(range(eval_n_iters)):
            try:
              batch, eval_ds_iter = load_batch(eval_ds, eval_ds_iter) # batch: [B, C, H, W]
            except ContinueIteration:
              print("continue invoked")
              continue
            mask = mask_init.detach().clone()[None, None].expand(*batch.shape[:2], -1, -1) # [B, C, H, W]
            print(f"x.shape: {batch.shape}, mask.shape: {mask.shape}")
            # masked_x = batch.masked_fill(mask, 0.0) # [B, C, H, W]
            original_x = batch.clone()
            # mean_prior = sde.config.training.mean_prior
            # sigma_prior = sde.config.training.sigma_prior
            # gaussian = torch.randn_like(batch) # [B, C, H, W]
            # gauss_noise = mean_prior + sigma_prior * gaussian.to(config.device)
            # masked_x = torch.where(mask, gauss_noise, batch)
            # masked_x = batch.where(mask, gauss_noise)
            single_sampling_iter(i, x=batch, mask=mask, grad_mask=mask, original_x=original_x)
        print("Done with inpainting. Exiting...")
        return
      else:
        print("Not using mask")
        for r in range(num_sampling_rounds):
          single_sampling_iter(r)
          

      # Compute inception scores, FIDs
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
        # print(stat)
        if not inceptionv3 and calculate_inception_score:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])

      if not inceptionv3 and calculate_inception_score:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]
      # Compute FID/IS on all samples together.
      if not inceptionv3 and calculate_inception_score:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e" % (
          ckpt, inception_score, fid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid)
        f.write(io_buffer.getvalue())