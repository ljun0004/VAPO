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
import ml_collections
def get_config():
  config = get_default_configs()
  data_dim = config.data.num_channels * config.data.image_size * config.data.image_size
  
  # training
  training = config.training
  training.sde = 'homotopy'
  training.batch_size = 128
  training.small_batch_size = 128
  training.num_particles = 1
  training.sample_size = 128
  training.sample_freq = 0
  training.mean_prior = 0
  training.sigma_prior = 1
  training.sigma_eps = 0
  training.mean_power = 1
  training.var_power = 1
  training.action_matching = True
  training.entropic = False
  training.coeff_diffusion = 0
  training.std_power_cov = 0.5
  training.std_power_vel = 0
  training.std_power_cvf = 0.5
  training.std_power_norm = 0
  training.std_power_temb = 0
  training.weight_cov = 1
  training.weight_vel = 0
  training.weight_cvf = 0.5
  training.weight_norm = 0.5
  training.weight_temb = 0.5
  training.divisor = math.sqrt(data_dim) * math.log(2*math.pi)
  # training.divisor = data_dim * math.log(2*math.pi)
  training.loss_multiplier = 1
  training.reduce_op = 'mean'
  training.sigma_min = 0.01
  training.sigma_max = 0.01
  training.eps_sigma = 1
  training.invert_sigma = False
  training.sigma_anneal = 0
  training.sigma_clip = 0.01
  training.t_start = 5e-8
  training.t_max = 1 - training.t_start
  training.t_end = 1 - training.t_start
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
  optim.laplacian_mode = 'flow'
  optim.cvf_mode = 'dot'
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.weight_decay = 0
  # optim.gamma = math.sqrt(2*math.pi)
  # optim.gamma = 2
  # optim.gamma = math.log(2*math.pi)
  # optim.gamma = math.sqrt(math.pi)
  # optim.gamma = math.sqrt(2)
  # optim.gamma = math.sqrt(math.log(2*math.pi))
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
  model.temb_type = 'lamb'
  model.name = 'unet'
  model.nonl = 'gelu'
  model.norm = 'none'
  model.bias = True
  model.data_init = False
  model.widen_factor = 1
  model.patch_size = 8
  model.dim = 512
  model.temb_dim = 128
  model.depth = 24
  model.sqr_freq = math.inf
  model.expansion_factor = (0,4)
  model.sqr_scale = 1
  model.skipmul_power = 1
  model.wrn_depth = 6 * 6 + 4
  model.wrn_width = 512 / 8
  model.ema_rate = 0.9999
  # model.normalization = 'GroupNorm'
  # model.nonlinearity = 'gelu'
  model.nf = 256
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.num_heads = 4
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.embedding_type = 'positional'
  model.conv_size = 3
  model.sigma_end = 0.01
  
  # data
  data = config.data
  data.dim = data_dim
  data.centered = True
  data.scale = 1
  data.num_classes = 10

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
  sampling.sigma = 0.0
  sampling.nfe = 100
  sampling.solver = 'RK45'
  sampling.normalize = False

  evaluate = config.eval
  evaluate.save_samples = False
  evaluate.save_images = False
  evaluate.classify_data_distribution=True
  evaluate.save_model_outs = False

 # data
  # config.eval.classify_dataset = csd = ml_collections.ConfigDict()
  # csd.eval.batch_size = 
  # csd.data = csdd =  ml_collections.ConfigDict()
  # csdd.dataset = 'CIFAR10'
  # csdd.channels = 3
  # csdd.image_size_ori = 32
  # csdd.image_size = 32
  # csdd.random_flip = True
  # csdd.centered = False
  # csdd.uniform_dequantization = False
  # csdd.num_channels = 3
  config.eval.classify_dataset = csd = deepcopy(svhn_configs())
  # config.eval.classify_dataset = csd = deepcopy(config)
  # config.eval.classify_dataset = csd = deepcopy(celeba_configs())
  csd.eval.batch_size = 100
  # csd.data.dataset = 'CELEBA'
  # csd.data.image_size_ori = 32
  # csd.data.image_size = 32
  # csd.data.random_flip = False
  # csd.data.centered = False
  # csd.data.uniform_dequantization = True
  # csd.data.num_channels = 3
  csd.data.dataset = 'SVHN'
  csd.data.image_size_ori = 32
  csd.data.image_size = 32
  # csd.data.random_flip = True
  # csd.data.centered = False
  # csd.data.uniform_dequantization = True
  # csd.data.num_channels = 3

  return config