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

# Lint as: python3
"""Training DDPM++ on CIFAR-10 with PFGM."""

from configs.default_cifar10_configs import get_default_configs
from configs.default_svhn_configs import get_default_configs as svhn_configs
from configs.default_celeba_configs import get_default_configs as celeba_configs
from copy import deepcopy
import math

def get_config():
  config = get_default_configs()
  data_dim = config.data.channels * config.data.image_size * config.data.image_size
  
  # training
  training = config.training
  training.sde = 'homotopy'
  training.batch_size = 64
  training.small_batch_size = 64
  training.num_particles = 1
  training.sample_size = 64
  training.sample_freq = 0
  training.mean_prior = 0
  training.sigma_prior = 1
  training.sigma_eps = 0
  training.mean_power = 1
  training.var_power = 1
  training.action_matching = True
  training.entropic = False
  training.coeff_diffusion = 0
  training.std_power_cov = 1
  training.std_power_vel = 0
  training.std_power_lap = 1
  training.std_power_norm = 0
  training.std_power_temb = 0
  training.weight_cov = 1
  training.weight_vel = 0
  training.weight_lap = 1
  training.weight_norm = 0.5
  training.weight_temb = 0.1
  # training.divisor = 1
  training.divisor = math.sqrt(data_dim) * math.log(2*math.pi)
  training.loss_multiplier = 1
  training.reduce_mean = True
  training.sigma_min = 0.01
  training.sigma_max = 0.01
  training.eps_sigma = 1
  training.invert_sigma = False
  training.sigma_anneal = 0
  training.sigma_clip = 0.01
  training.t_start = 1e-5
  training.t_max = 1 - training.t_start
  training.t_end = 1
  training.scheduling = 'fmot'
  training.shift_schedule = False
  training.shift_ref = 64
  training.eps_max = 1e-4
  training.eps_min = 1e-4
  training.continuous = True
  training.invert_t = False
  training.augment_t = True
  training.z_max = 1
  training.eps_z = 1
  training.augment_z = False
  training.model = 'ddpmpp'
  training.method = 'posterior'
  training.class_guidance = False
  training.p_uncond = 0.2
  training.snapshot_freq = 10000
  training.snapshot_freq_for_preemption = 10000

  # optim
  optim = config.optim
  optim.cov = True
  optim.vel = False
  optim.norm = True
  optim.laplacian = True
  optim.laplacian_mode = 'fmot'
  optim.cov_mode = 'corr'
  optim.cvf_mode = 'cos'
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.weight_decay = 0
  optim.alpha = 0
  optim.delta = 0
  optim.beta1 = 0.9
  optim.grad_clip = 3
  optim.grad_clip_mode = 'std'
  optim.warmup = 5000
  
  # model
  model = config.model
  model.multiplier = 0.5
  model.divisor = 1
  model.scale_by_sigma = False
  model.temb_type = 'time'
  model.name = 'unet'
  model.nonl = 'silu'
  model.norm = 'none'
  model.bias = True
  model.data_init = False
  model.ema_rate = 0.9999
  model.nf = 128
  model.num_res_blocks = 4
  model.attention_resolutions = [2]
  model.dropout = 0.3
  model.channel_mult = [2, 2, 2]
  model.conv_resample = False
  model.dims = 2
  model.num_classes = None
  model.use_checkpoint = False
  model.num_heads = 1
  model.num_head_channels = -1
  model.num_heads_upsample = -1
  model.use_scale_shift_norm = True
  model.resblock_updown = False
  model.use_new_attention_order = True
  model.with_fourier_features = False

  # data
  data = config.data
  data.dim = data_dim
  data.centered = True
  data.scale = 1
  data.classes = 10

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  sampling.ode_solver = 'rk45'
  sampling.eps_z = 1e-10
  #sampling.ode_solver = 'forward_euler'
  #sampling.ode_solver = 'improved_euler'
  sampling.N = 100
  sampling.z_max = 40
  sampling.z_min = 1e-3
  sampling.upper_norm = 3000
  # verbose
  sampling.vs = False
  sampling.guidance_strength = 3
  ## new args
  sampling.t_end = training.t_max
  sampling.rtol = 1e-3
  sampling.atol = 1e-3
  sampling.eps_z = 1e-10
  sampling.weight = 1.0
  sampling.nfe = 100
  sampling.solver = 'RK45'

  evaluate = config.eval
  evaluate.save_samples = False
  evaluate.save_images = False
  evaluate.classify_data_distribution=True
  evaluate.save_model_outs = False

  config.eval.classify_dataset = csd = deepcopy(svhn_configs())
  csd.eval.batch_size = 100
  csd.data.dataset = 'SVHN'
  csd.data.image_size_ori = 32
  csd.data.image_size = 32
  

  return config