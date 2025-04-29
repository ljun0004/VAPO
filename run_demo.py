#%% main.py
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

class flags:
    def __init__(self, workdir=None, mode=None):
        
        from configs.homotopy.cifar10_silu import get_config
        # from configs.homotopy.celeba import get_config

        self.config = get_config()
        self.workdir = workdir
        self.mode = mode
        # self.eval_folder = "eval"

save_dir = "/home/admin01/Junn/VAPO-FM/cifar10_FM_mean_corr_cos_sc1_sl1_wc1_wl1_wn0.5_wt0.1_ts1e-5_Adam_silu"

FLAGS = flags(workdir=save_dir, mode="eval")
tf.io.gfile.makedirs(FLAGS.workdir)
# Set logger so that it outputs to both console and file
# Make logging work for both disk and Google Cloud Storage
gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
handler = logging.StreamHandler(gfile_stream)
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel('INFO')

## %% run_lib.py
import gc
import io
import os
import time
import copy

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
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import datasets_utils.celeba

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

#%% run_lib.py - train
config = FLAGS.config
workdir = FLAGS.workdir
# config.device = torch.device('cpu')

# Create directories for experimental logs
sample_dir = os.path.join(workdir, "samples")
tf.io.gfile.makedirs(sample_dir)

tb_dir = os.path.join(workdir, "tensorboard")
tf.io.gfile.makedirs(tb_dir)
writer = tensorboard.SummaryWriter(tb_dir)

# Initialize model.
net = mutils.create_model(config)
print("Number of model parameters: %.5e" %sum(p.numel() for p in net.parameters()))
ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
optimizer = losses.get_optimizer(config, net.parameters())
print("Weight decay: %.5f" %optimizer.param_groups[0]['weight_decay'])
state = dict(optimizer=optimizer, model=net, ema=ema, step=0, sigma_max=config.training.sigma_max, t_eval=0)

# Create checkpoints directory
print(workdir)
checkpoint_dir = os.path.join(workdir, "checkpoints")
# Intermediate checkpoints to resume training after pre-emption in cloud environments
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "cifar10.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "cifar10_sc2.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "cifar10_te2_390000.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "celeba.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "celeba_te1_100000.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "celeba_te1_270000.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "celeba_te2.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "celeba_te2_100000.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "celeba_te2_270000.pth")
checkpoint_meta_dir = os.path.join(workdir, "checkpoints", "checkpoint_1090000.pth")
# checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
tf.io.gfile.makedirs(checkpoint_dir)
tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
# Resume training when intermediate checkpoints are detected
state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
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
  sampling_eps = config.sampling.eps_z
else:
  raise NotImplementedError(f"Method {config.training.sde} unknown.")

# Build one-step training and evaluation functions
optimize_fn = losses.optimization_manager(config)
reduce_mean = config.training.reduce_mean
method_name = config.training.sde.lower()
# train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
#                                     reduce_mean=reduce_mean, method_name=method_name)
# eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
#                                   reduce_mean=reduce_mean, method_name=method_name)

# Building sampling functions
if config.training.snapshot_sampling:
  sampling_shape = (25, config.data.channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

num_train_steps = config.training.n_iters

# In case there are multiple hosts (e.g., TPU pods), only log to host 0
# logging.info("Starting training loop at step %d." % (initial_step,))
print(initial_step)
for step in range(initial_step, num_train_steps + 1):
  break
# Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
if config.data.dataset == 'CELEBA':
  try:
    batch = next(train_iter)[0].cuda()
  except StopIteration:
    train_iter = iter(train_ds)
    batch = next(train_iter)[0].cuda()
else:
  batch_ds = next(train_iter)
  batch = torch.from_numpy(batch_ds['image']._numpy()).to(config.device).float()
  batch = batch.permute(0, 3, 1, 2)
  # JULIA's CEMB change Get labels and do one-hot encoding
  labels = torch.from_numpy(batch_ds['label']._numpy()).to(config.device).long()
batch = scaler(batch)

# # %% losses.py
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# from scipy import integrate
# import math
# import time
# from models import utils as mutils
# from methods import VESDE, VPSDE
# from models import utils_poisson

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# from models.ebm_models import get_timestep_embedding
# from models.ebm_models import GaussianFourierProjection
# from sklearn.preprocessing import OneHotEncoder
# from einops import rearrange

# scaler = datasets.get_data_scaler(config)
# inverse_scaler = datasets.get_data_inverse_scaler(config)

# # Execute one training step
# model = state['model']
# optimizer = state['optimizer']
# # optimizer.zero_grad()
# train = True
# continuous = True
# eps = 1e-5
# sample_bool = False
# ind_bool = True
# torch.manual_seed(49)

# # Get configs
# data_dim = sde.config.data.dim
# batch_size = sde.config.training.batch_size
# ensemble_size = sde.config.training.small_batch_size
# if not ind_bool:
#   assert batch_size == ensemble_size
# # num_particles = sde.config.training.num_particles
# sample_size = sde.config.training.sample_size

# # Get the mini-batch
# # print(batch.shape)
# batch = batch.reshape(batch_size, -1)
# samples_batch = batch[:ensemble_size, :data_dim]
# device = batch.device

# # Get one-hot label encodings
# if sde.config.training.class_guidance:
#   # labels_nested = [[label] for label in labels]m
#   # encoder = OneHotEncoder(categories='auto', sparse_output=False)
#   # labels = encoder.fit_transform(labels_nested)
#   # labels = torch.from_numpy(labels).to(device)
#   labels = F.one_hot(labels, num_classes=sde.config.data.classes)
#   # logging.info(labels.shape)
#   # ----labels shape torch.Size([batch_size, num of labels])
#   # Random 
#   random = torch.rand(labels.shape[0]).to(device)
#   # Should probably turn this into an arg
#   p_uncond = sde.config.training.p_uncond
#   uncond_mask = (random < p_uncond)[:,None]
#   # What should null be -> [0,0,0,0,0,0,0,0,0,0]? 
#   null = torch.zeros(labels.shape[-1]).to(device)
#   labels =  torch.where(uncond_mask, null, labels)
#   # labels = labels_null

# # Get prior sigma
# # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels)
# # sigma_prior = math.sqrt(sde.config.data.image_size) * sde.config.data.channels
# # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.channels)
# mean_prior = sde.config.training.mean_prior
# sigma_prior = sde.config.training.sigma_prior

# # Change-of-variable: z = -ln(t)
# z_t = lambda t: -torch.log(t)
# t_z = lambda z: torch.exp(-z)

# t_start = 1e-5
# t_max = 1 - t_start
# t_end = 2

# with torch.no_grad():
  
#   # Sample t from log-uniform distribution
#   t_range = (t_start, t_end)
#   t_max = t_max

#   # Noise scheduling
#   if sde.config.training.continuous:
#     t_samples = t_range[0] + torch.rand(ensemble_size).to(device) * (t_range[1] - t_range[0])
#   else:
#     t_samples = t_range[0] + (torch.randint(0, 1000, (ensemble_size, 1)).to(device) / 1000) * (t_range[1] - t_range[0]) 
#   t_clipped = t_samples.clone()
#   t_clipped[t_samples > t_max] = t_max
#   t_samples = t_samples[:,None]
#   t_clipped = t_clipped[:,None]

#   # s = torch.distributions.Exponential(rate=1.0).sample([ensemble_size, 1]).to(device)
#   # s_max = 100
#   # s = torch.clamp(s, max=s_max)
#   # t_samples = 1 - torch.exp(-s)
#   # t_clipped = t_samples.clone()

#   # plt.plot(s.cpu(), s.cpu(), '.')
#   # plt.show()
#   # plt.plot(t_clipped.cpu(), t_clipped.cpu(), '.')
#   # plt.show()

#   mean_dt = 1 * torch.ones_like(t_clipped)
#   std_dt = -1 * torch.ones_like(t_clipped)
#   mean_enc_batch = t_clipped
#   var_enc = (1 - t_clipped)

#   # Compute enc (posterior) mean and var
#   # sigma_dec = sde.config.training.sigma_min
#   # var_dec = sigma_dec**2
#   # tao_samples = torch.exp(lamb_samples) * var_dec
#   # var_prior = sigma_prior**2
#   # mean_enc_batch = (tao_samples * var_prior) / (var_dec + tao_samples * var_prior)
#   # # mean_enc_prior = var_dec / (var_dec + tao_samples * var_prior)
#   # var_enc = var_prior * var_dec / (var_dec + tao_samples * var_prior)

#   # Perturb data samples with gaussians
#   gaussians_x = torch.randn(ensemble_size, data_dim).to(device)
#   mean_enc = mean_enc_batch.pow(sde.config.training.mean_power) * samples_batch
#   std_enc = var_enc.pow(sde.config.training.var_power)
#   samples_x = mean_enc + (std_enc + sde.config.training.sigma_eps) * gaussians_x
#   const = data_dim * var_enc + (1 - mean_enc_batch).pow(2) * samples_batch.pow(2).sum(dim=-1, keepdim=True)

#   if sample_bool:
#     samples_s, nfe = sampling_fn(model, state, sample_size=sample_size, method='RK23', eps=1e-3, rtol=1e-3, atol=1e-3, inverse_scale=False)
#     print("step: %d, nfe: %d" % (state['step'], nfe))
#     # print(samples_x.min(), samples_x.max(), samples_s.min(), samples_s.max())
#     samples_x[:sample_size] = samples_s

# # with torch.enable_grad():
# #   # Get model function
# #   net_fn = mutils.get_predict_fn(sde, model, train=train, continuous=continuous)

# #   # Predict scalar potential
# #   samples_x.requires_grad = True
# #   # time embeddings
# #   if sde.config.training.augment_t:
# #     if sde.config.model.temb_type == 'time':
# #       cond_samples = t_samples
# #     elif sde.config.model.temb_type == 'lamb': 
# #       lamb_samples = 1 * torch.log((t_clipped) / (1-t_clipped))
# #       cond_samples = lamb_samples
# #     if sde.config.training.action_matching:
# #       cond_samples.requires_grad = True
# #       samples_net = torch.cat([samples_x, cond_samples, std_enc], dim=-1)
# #     else:
# #       temb = get_timestep_embedding(cond_samples.squeeze(dim=-1), sde.config.model.nf)  
# #       # temb = GaussianFourierProjection(sde.config.model.nf, emb_type='fourier')(cond_samples)
# #       temb.requires_grad = True
# #       samples_net = torch.cat([samples_x, temb], dim=-1)
# #   else:
# #     samples_net = samples_x
# #   psi = net_fn(samples_net).squeeze(dim=-1)
# #   # print(psi.shape)

# #   # Normalize potential by its mean
# #   # psi -= psi.mean(dim=0, keepdim=True)

# #   # Compute (backpropagate) N-dimensional Poisson field (gradient)
# #   if sde.config.training.augment_t:
# #     if sde.config.training.action_matching:
# #       drift_x, drift_emb = torch.autograd.grad(psi, [samples_x, cond_samples], torch.ones_like(psi), create_graph=True)
# #     else:
# #       drift_x, drift_emb = torch.autograd.grad(psi, [samples_x, temb], torch.ones_like(psi), create_graph=True)
# #   else:
# #     drift_x = torch.autograd.grad(psi, samples_x, torch.ones_like(psi), create_graph=True)[0]
# #   laplacian_x = torch.autograd.grad(drift_x, samples_x, torch.ones_like(drift_x), create_graph=True)[0]

# # # Compute drift norm
# # if sde.config.training.augment_t: 
# #   Norm_emb = drift_emb.pow(2) * (1 + t_samples).pow(sde.config.training.std_power_temb)
# #   Norm_emb = Norm_emb.sum(dim=-1)

# # if sde.config.optim.cvf:
# #   if sde.config.training.score_matching:
# #     vf_cond = (std_dt * gaussians_x)
# #   else:
# #     vf_cond = (mean_dt * batch) + (std_dt * gaussians_x)
    
# #   if sde.config.optim.cvf_mode == 'dist':
# #     Norm_cvf = (drift_x - vf_cond).pow(2).sum(dim=-1)
# #   elif sde.config.optim.cvf_mode == 'dot':
# #     Norm_cvf = - (drift_x * vf_cond).sum(dim=-1)
# #   elif sde.config.optim.cvf_mode == 'cos':
# #     Norm_cvf = - F.cosine_similarity(drift_x, vf_cond, dim=-1)
  
# # if sde.config.training.entropic:
# #   etp_cond = - (gaussians_x / (1 - t_clipped)) * (sde.config.training.coeff_diffusion)**2
# #   Norm_etp = (drift_x - etp_cond).pow(2) * (1 + t_samples).pow(sde.config.training.std_power_norm)
# # else:
# #   Norm_etp = drift_x.pow(2) * (1 + t_samples).pow(sde.config.training.std_power_norm)
# # Norm_etp = Norm_etp.sum(dim=-1)
  
# # Norm = sde.config.training.weight_norm * Norm_etp
# # if sde.config.training.augment_t:
# #   Norm += sde.config.training.weight_temb * Norm_emb
# # if sde.config.optim.cvf:
# #   Norm += sde.config.training.weight_cvf * Norm_cvf
  
# # Laplacian = 0.5 * laplacian_x * (1 + t_samples).pow(sde.config.training.std_power_laplacian)
# # Laplacian = Laplacian.sum(dim=-1)

# if sde.config.training.method == 'linear':

#   with torch.no_grad():
#     # Compute Normalized Innovation Squared (Gamma)
#     if ind_bool:
#       distance = batch.unsqueeze(dim=1) - samples_x
#     else:
#       distance = batch - samples_x

#     innovation = distance.pow(2).sum(dim=-1)
#     # innovation = innovation.sqrt()
    
#     if ind_bool:
#       Gamma = innovation.mean(dim=0)
#       Gamma -= Gamma.mean(dim=0, keepdim=True)
#     else:
#       Gamma = innovation - const.mean(dim=0)
  
# elif sde.config.training.method == 'posterior':  

#   with torch.no_grad():
#     # Compute Normalized Innovation Squared (Gamma)
#     innovation = gaussians_x.pow(2) - (batch * gaussians_x)
#     innovation = innovation * (1 - t_clipped).pow(2)
#     Gamma = innovation.sum(dim=-1)
#     Gamma = Gamma - Gamma.mean()

# # Compute sample correlation between potential and NIS
# divisor = data_dim * math.log(2*math.pi)
# # Gamma = Gamma / (divisor + eps)
# # Cov = Gamma * psi

# plt.plot(t_samples.squeeze().cpu(), Gamma.squeeze().cpu(), 'r.')
# plt.show()
# plt.plot(Gamma.squeeze().cpu(), 0*np.ones_like(Gamma.squeeze().cpu()), 'r.')
# plt.show()
# print(Gamma.max() - Gamma.min(), Gamma.std())
  
# # Vel = - drift_emb * (1 - t_clipped).pow(sde.config.training.std_power_vel)
# # Vel = Vel.sum(dim=-1)

# # Cov = Cov.mean(dim=0)
# # Loss = torch.zeros_like(Cov)
# # # print(Cov.shape)
# # Loss += sde.config.training.weight_cov * Cov

# # Vel = Vel.mean(dim=0)
# # # print(Vel.shape)
# # Loss += sde.config.training.weight_vel * Vel

# # Norm = Norm.mean(dim=0)
# # # print(Norm.shape)
# # Loss += Norm

# # Laplacian = Laplacian.mean(dim=0)
# # # print(Laplacian.shape)
# # Loss += sde.config.training.weight_laplacian * Laplacian

# # # Reg = drift_x.pow(2).sum(dim=-1)
# # Reg = Norm_etp.mean(dim=0)
# # if sde.config.optim.alpha > 0:
# #   Loss += sde.config.optim.alpha * Reg

# # if sde.config.model.name == 'vaebmwrn':
# #   Spec = model.module.spectral_norm_parallel()
# #   # print(Spec)
# #   Loss += sde.config.optim.delta * Spec

# # # Nll = cvf.sum(dim=-1)
# # Nll = Norm_cvf.mean(dim=0)

# # # print(psi_0.mean().item(), psi_1.mean().item(), drift_emb.mean().item())

# # if sde.config.training.reduce_op == 'mean':
# #   Loss = sde.config.training.loss_multiplier * Loss / data_dim
# # else:
# #   Loss = sde.config.training.loss_multiplier * Loss
# # # print(Loss.shape)

# # sample = inverse_scaler(batch.reshape(-1,config.data.channels, config.data.image_size, config.data.image_size))
# sample = inverse_scaler(samples_x[:,:data_dim]).reshape(-1,config.data.channels, config.data.image_size, config.data.image_size).detach()
# nrow = int(np.sqrt(sample.shape[0]))
# image_grid = make_grid(sample, nrow, padding=2)
# sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
# save_path = './sample_batch.png'
# with tf.io.gfile.GFile(save_path, "wb") as fout:
#   save_image(image_grid, fout)
# img = plt.imread(save_path) 
# plt.imshow(img)
# plt.show()

# #%% scheduling
# import torch
# import math
# import matplotlib.pyplot as plt
# from models.ebm_models import get_timestep_embedding
# from models.ebm_models import GaussianFourierProjection

# sigma_prior = 1
# sigma_min = 0.01

# # var_dec = sigma_min**2
# # t_samples = torch.linspace(0,1,100)
# # eps = 1e-4
# # t_log = torch.linspace(math.log(eps),math.log(1+eps),100)
# # t_samples = torch.exp(t_log) - eps
# # var_prior = sigma_prior**2
# # var_dec = sigma_min**2
# # # mean_enc = (t_samples * var_prior + mean_prior * var_dec) / (var_dec + t_samples * var_prior)
# # mean_enc_y = (t_samples * var_prior) / (var_dec + t_samples * var_prior)
# # mean_enc_prior = var_dec / (var_dec + t_samples * var_prior)
# # var_enc = var_prior * var_dec / (var_dec + t_samples * var_prior)

# # plt.plot(t_log, mean_enc_y, t_log, var_enc)
# # plt.show()

# ts = 1e-2
# var_dec = ts * sigma_min**2
# t_samples = torch.linspace(ts,1-ts,100)
# # t_samples = (t_range[0] + torch.linspace(0,1000,1001)/1000) * (t_range[1] - t_range[0]) 

# # fmot
# kappa_samples = (t_samples)**1/(1-t_samples)**1
# lamb_samples = torch.log(kappa_samples)
# # lamb_samples += math.log(512 / sde.config.data.image_size)
# tao_samples = var_dec * torch.exp(lamb_samples)
# # slope = 1/(t_samples*(1-t_samples))
# # slope = -0.5/(1-t_samples)**(0.5)
# # slope = ((1)**2) / (1-t_samples)
# slope = (1-t_samples)**(-1)
# # slope = var_dec * slope
# # Compute enc (posterior) mean and var
# var_prior = sigma_prior**2
# mean_enc_y = (tao_samples * var_prior) / (var_dec + tao_samples * var_prior)
# mean_enc_prior = var_dec / (var_dec + tao_samples * var_prior)
# var_enc = var_prior * var_dec / (var_dec + tao_samples * var_prior)

# # plt.plot(t_samples, kappa_samples)
# # plt.show()
# plt.plot(t_samples, lamb_samples)
# plt.show()
# # plt.plot(t_samples, tao_samples)
# # plt.show()
# plt.plot(t_samples, slope)
# plt.show()
# plt.plot(t_samples, mean_enc_y**1, 'r--')
# plt.plot(t_samples, var_enc**1, 'r-.')
# # plt.show()
# # plt.plot(t_samples, 1*torch.ones_like(t_samples), 'b')
# # plt.plot(t_samples, -1*torch.ones_like(t_samples), 'b')
# plt.show()
# print(slope)

# print(min(t_samples),max(t_samples))
# print(min(lamb_samples),max(lamb_samples))
# print(min(tao_samples),max(tao_samples))
# print(min(slope),max(slope))
# print(min(mean_enc_y),max(mean_enc_y))
# print(min(var_enc),max(var_enc))

# # cosine
# kappa_samples = torch.tan(math.pi*t_samples/2)**2
# # kappa_samples = (t_samples)**2/(1-t_samples)**2
# lamb_samples = torch.log(kappa_samples)
# # lamb_samples += math.log(512 / sde.config.data.image_size)
# tao_samples = var_dec * torch.exp(lamb_samples)
# # slope = 1/(1-t_samples)**(1)
# # slope = var_dec * slope
# # Compute enc (posterior) mean and var
# var_prior = sigma_prior**2
# mean_enc_y = (tao_samples * var_prior) / (var_dec + tao_samples * var_prior)
# mean_enc_prior = var_dec / (var_dec + tao_samples * var_prior)
# var_enc = var_prior * var_dec / (var_dec + tao_samples * var_prior)

# # plt.plot(t_samples, kappa_samples)
# # plt.show()
# # plt.plot(t_samples, lamb_samples)
# # plt.show()
# # plt.plot(t_samples, tao_samples)
# # plt.show()
# # plt.plot(t_samples, slope)
# # plt.show()
# # plt.plot(t_samples, mean_enc_y**1, 'r--')
# # plt.plot(t_samples, var_enc**1, 'r-.')
# # plt.show()
# # plt.plot(t_samples, 1*torch.ones_like(t_samples), 'b')
# # plt.plot(t_samples, -1*torch.ones_like(t_samples), 'b')
# # plt.show()

# # vapo
# # kappa_samples = torch.tan(math.pi*t_samples/2)**2
# kappa_samples = (t_samples)**2/(1-t_samples)**2
# lamb_samples = torch.log(kappa_samples)
# # lamb_samples += math.log(512 / sde.config.data.image_size)
# tao_samples = var_dec * torch.exp(lamb_samples)
# # slope = 1/(1-t_samples)**(1)
# # slope = var_dec * slope
# # Compute enc (posterior) mean and var
# var_prior = sigma_prior**2
# mean_enc_y = (tao_samples * var_prior) / (var_dec + tao_samples * var_prior)
# mean_enc_prior = var_dec / (var_dec + tao_samples * var_prior)
# var_enc = var_prior * var_dec / (var_dec + tao_samples * var_prior)

# # plt.plot(t_samples, kappa_samples)
# # plt.show()
# # plt.plot(t_samples, lamb_samples)
# # plt.show()
# # plt.plot(t_samples, tao_samples)
# # plt.show()
# # plt.plot(t_samples, slope)
# # plt.show()
# # plt.plot(t_samples, mean_enc_y**1, 'r')
# # plt.plot(t_samples, var_enc**1, 'r')
# # plt.show()
# # plt.plot(t_samples, 1*torch.ones_like(t_samples), 'b')
# # plt.plot(t_samples, -1*torch.ones_like(t_samples), 'b')
# # plt.show()

# # # ori
# # lamb_samples = -1 * torch.log((1-t_samples)**2/(t_samples**2))
# # lamb_samples += math.log(64 / sde.config.data.image_size)
# # tao_samples = var_dec * torch.exp(lamb_samples)
# # # Compute enc (posterior) mean and var
# # var_prior = sigma_prior**2
# # mean_enc_y = (tao_samples * var_prior) / (var_dec + tao_samples * var_prior)
# # mean_enc_prior = var_dec / (var_dec + tao_samples * var_prior)
# # var_enc = var_prior * var_dec / (var_dec + tao_samples * var_prior)
# # # plt.plot(t_samples, lamb_samples)
# # # plt.show()
# # # plt.plot(t_samples, tao_samples)
# # # plt.show()
# # # plt.plot(t_samples, slope)
# # # plt.show()
# # plt.plot(t_samples, mean_enc_y**0.5, 'k')
# # plt.plot(t_samples, var_enc**0.5, 'k')

# # plt.legend(['fmot','vapo','elbo','ori']) 
# # plt.show()

# # # # temb_projection = get_timestep_embedding
# # # temb_projection = GaussianFourierProjection()

# # # T = 1000
# # # t_samples = np.ceil(t_samples * T) / T
# # # temb = temb_projection(t_samples)
# # # plt.plot(t_samples, temb.mean(-1))
# # # plt.show()
# # # temb = temb_projection(torch.log(t_samples))
# # # plt.plot(t_samples, temb.mean(-1))
# # # plt.show()
# # # temb = temb_projection(lamb_samples)
# # # plt.plot(t_samples, temb.mean(-1))
# # # plt.show()
# # # temb = temb_projection(torch.log(lamb_samples))
# # # plt.plot(t_samples, temb.mean(-1))
# # # plt.show()
# # # print(t_samples)

# %% sampling.py
import functools
import torch
import numpy as np
import abc
import time
import math

from models.utils import from_flattened_numpy, to_flattened_numpy, get_predict_fn
from scipy import integrate
import methods
from models import utils as mutils
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid, save_image

from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy.spatial import geometric_slerp
from mask import get_masked_fn
# from models.ebm_models import get_timestep_embedding
# from models.ebm_models import GaussianFourierProjection

# torch.cuda.empty_cache()
model = state['model']
ema.store(model.parameters())
ema.copy_to(model.parameters())

data_dim = sde.config.data.channels * sde.config.data.image_size * sde.config.data.image_size
if config.training.snapshot_sampling:
  sampling_shape = (4, config.data.channels, config.data.image_size, config.data.image_size)
shape = sampling_shape
sample_size = shape[0]
device = config.device
x = None
mask = None
grad_mask = None
# method='RK23'
# method='RK45'
method='Euler'
# method='DOP853'
rtol = 1e-3
atol = 1e-3
eps_z = 1e-8
c = 1
step_size = 0
torch.manual_seed(49)
inpainting = False
interpolation = False
longrun = True
c_norm = 1
dcoeff = 0.35 #0.35 celeba #0.3 cifar10
t_start = 0
t_max = 1
t_end = 20
nplot = 16
fps = 50
nfe = fps * t_end
f = - (1 / t_max)
# g = (2 * (1 - t_max) / t_max)**(0.5)
g = 1
print(t_start, t_max, t_end, f, g)
mode = 'time'
# weight = 1.075
weight = 0.5

if inpainting:
  batch = batch[:sample_size]
  t_sample = 2.5e-5
else:
  # batch = batch[:sample_size]
  batch = None
  t_sample = 0

# save_path_sample = './cifar10_small_8.png'
# save_path_sample = './cifar10_big.png' 
# save_path_sample = './cifar10_interp_big.png'
# save_path_sample = './celeba_small_2.png'
# save_path_sample = './celeba_big_2.png'
# save_path_sample = './celeba_interp_small.png'
# save_path_sample = './celeba_interp_big.png'
  
# save_path_sample = './cifar10_'+str(method)+'_'+str(mode)+'_'+str(t_end)+'.png'
save_path_sample = './cifar10_'+str(method)+'_'+str(mode)+'_'+str(t_end)+'_'+str(nfe)+'_'+'.png'
# save_path_sample = './celeba_'+str(method)+'_'+str(mode)+'_'+str(t_end)+'_'+str(nfe)+'_'+'.png'

with torch.no_grad():
  data_dim = sde.config.data.channels * sde.config.data.image_size * sde.config.data.image_size
  # batch_size = sde.config.training.batch_size
  # ensemble_size = sde.config.training.small_batch_size
  # num_particles = sde.config.training.num_particles
  # var_dec = sde.config.training.sigma_min**2

  # Change-of-variable: z = -ln(t)
  z_t = lambda t: -c * math.log(t)
  t_z = lambda z: math.exp(-z/c)
  l_t = lambda t: math.log(t/(1-t))
  t_l = lambda l: 1/(1 + math.exp(-l))

  # boundary = [0, t_end]
  # boundary = [z_t(eps_z), z_t(t_end)]
  if mode == 'time':
    boundary = [t_start, t_end]
    ubound = t_max
  elif mode == 'lamb':
    boundary = [l_t(t_start), l_t(t_end)]
    ubound = l_t(t_start)
  # boundary = [np.log(sde.config.training.z_max), np.log(eps)]

  # Initial sample
  if x is None:
    # Geometric sequence of sigmas
    # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels)
    # sigma_prior = math.sqrt(sde.config.data.image_size) * sde.config.data.channels
    # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.channels)
    mean_prior = sde.config.training.mean_prior
    sigma_prior = sde.config.training.sigma_prior
    sigma_dec = sde.config.training.sigma_min

    var_prior = sigma_prior**2
    var_dec = sigma_dec**2

    # Sample from prior
    mean_enc_y = (t_sample * var_prior) / (var_dec + t_sample * var_prior)
    mean_enc_prior = var_dec / (var_dec + t_sample * var_prior)
    var_enc = var_prior * var_dec / (var_dec + t_sample * var_prior)

    if inpainting:
      config.eval.mask = True
      config.eval.mask_type = "box_center"
      config.eval.mask_box_size = 32
      mask_fn = get_masked_fn(config)
      mask_init = mask_fn(config).bool() # [H, W]
      mask = mask_init.detach().clone()[None, None].expand(*batch.shape[:2], -1, -1) # [1, 1, H, W]
      # mask = ~mask
      # grad_mask = mask.reshape(len(mask), -1).to(device) 
      masked_x = batch.masked_fill(mask, 0) # [B, C, H, W]
      # x = masked_x.to(device)
      gaussian = torch.randn(sample_size, data_dim).to(device)
      x = math.sqrt(var_enc) * gaussian
      if batch is not None:
        x += mean_enc_y * masked_x.reshape(len(masked_x), -1).to(device)
        x += mean_enc_prior * mean_prior
      # x = torch.where(mask, gaussian.reshape(sampling_shape), x.reshape(sampling_shape))
      x = x.reshape(len(x), -1).to(device) 

    elif interpolation:
      gaussian = torch.randn(sample_size, data_dim)
      x_init = math.sqrt(var_enc) * gaussian.to(device)
      n_interp = 20
      t_vals = np.linspace(0, 1, n_interp)
      unit_vec = x_init.detach().cpu().numpy().astype(np.double)
      unit_mag = np.sqrt(np.sum(unit_vec**2, axis=-1, keepdims=True))
      unit_vec /= unit_mag
      x = []
      for i in range(round(sample_size // n_interp)):
        interp = geometric_slerp(unit_vec[i], unit_vec[i+1], t_vals)
        x.append(torch.from_numpy(interp))
      x = torch.stack(x).to(device) * unit_mag.mean()

    else:
      gaussian = torch.randn(sample_size, data_dim)
      x =  math.sqrt(var_enc) * gaussian.to(device)
      if batch is not None:
        x += mean_enc_y * batch.reshape(len(batch), -1).to(device)
        # x = 1 * x + 0 * batch.reshape(len(batch), -1).to(device)
        # .mean(axis=0, keepdims=True)
        x += mean_enc_prior * mean_prior
      norm_x = torch.sqrt(torch.sum(x**2, axis=1, keepdims=True))
      x = (x / norm_x) * c_norm * norm_x.mean()

    x_prior = x.clone()

    if sde.config.training.augment_z: 
      x = torch.cat([x, z_t(eps_z) * torch.ones(sample_size, 1).to(device)], dim=-1)

  # x = x.view(shape).float()
  new_shape = (sample_size, sde.config.data.channels, sde.config.data.image_size, sde.config.data.image_size)

  # t = np.log(sde.config.sampling.z_max)
  # x = to_flattened_numpy(x)
  state['t_eval'] = 0
  Time = []
  Energy = []
  Norm = []
  X_samples = []
  plot_idx = 0
  time_eval = np.linspace(t_start, t_end, nplot)
  start_time = time.time()
  
  if method == 'Euler':
##================================================================================================================##
    for i, z in enumerate(np.linspace(boundary[0], boundary[1], nfe+1)):
    # for t in (1 - np.linspace(0, 1, nfe+1) ** 1)[::-1]:

      step_size =  abs(z - state['t_eval'])
      state['t_eval'] = z

      # Prepare potential network input
      if mode == 'time':
        if z > ubound:
          t = ubound
        else:
          t = z
        # lamb = l_t(t)
      elif mode == 'lamb':
        if z > ubound:
          lamb = ubound
        else:
          lamb = z
        # t = t_l(lamb)
      
      if sde.config.training.augment_t:
        if sde.config.model.temb_type == 'lamb': 
          if sde.config.training.scheduling == 'cosine':
            lamb = 2 * math.log(math.tan(t*math.pi/2))
          elif sde.config.training.scheduling == 'fmot':
            lamb = -1 * math.log((1-t)**1/(t)**1)
          elif sde.config.training.scheduling == 'vapo':
            lamb = -1 * math.log((1-t)**2/(t)**1)
          elif sde.config.training.scheduling == 'elbo':
            lamb = -1 * math.log((1-t)**2/(2*t**2))
          if sde.config.training.shift_schedule:
            lamb += math.log(64 / sde.config.data.image_size)
      
      # state['t_eval'] += 1; step_size = abs(t) / state['t_eval']

      samples_x = x
      # samples_x[:,:-1] += torch.randn_like(samples_x[:,:-1]) * math.sqrt(0.01)
      if sde.config.training.augment_z: 
        z = z_t(t)
        samples_x = torch.cat([samples_x[:,:-1], z * torch.ones(sample_size, 1).to(device).type(torch.float32)], dim=-1)
      samples_x.requires_grad = True

      with torch.enable_grad():
        # Get model function
        net_fn = get_predict_fn(sde, model, train=False)

        # Predict scalar potential (FC)
        # samples_net = samples_x
        if sde.config.training.augment_t:
          if sde.config.model.temb_type == 'time':
            cond = t
          elif sde.config.model.temb_type == 'lamb': 
            cond = lamb
          cond_samples = cond * torch.ones(sample_size, 1).to(device).type(torch.float32)
          std_enc = (1-t) * torch.ones(sample_size, 1).to(device).type(torch.float32)
          # temb = GaussianFourierProjection(sde.config.model.temb_dim, emb_type='fourier')(cond_samples)
          samples_net = torch.cat([samples_x, cond_samples, std_enc], dim=-1)
        if sde.config.training.class_guidance:
          # 1-airplane  2-automobile  3-bird  4-cat  5-deer  6-dog  7-frog  8-horse  9-ship  10-truck
          # labels = torch.tensor([0,0,0,1,0,0,0,0,0,0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
          # samples_net = torch.cat([samples_x, cond_samples, labels], dim=-1)
          labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
          samples_net = torch.cat([samples_x, cond_samples, std_enc, labels], dim=-1)
          
        psi = net_fn(samples_net).squeeze(dim=-1)
        # if sde.config.training.class_guidance:
        #   psi_null = net_fn(samples_net_null).squeeze(dim=-1)

        # Normalize field by its mean
        # psi -= psi.mean(dim=0, keepdim=True)

        # Compute (backpropagate) N-dimensional Poisson field (gradient)
        drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi))[0]
        # if sde.config.training.class_guidance:
        #   drift_null = torch.autograd.grad(psi_null, samples_x, torch.ones_like(psi))[0]
        #   w = sde.config.sampling.guidance_strength
        #   drift = (1 + w) * drift - w * drift_null

        if mode == 'lamb':
          dt_dl = t*(1-t)
          drift = drift * dt_dl

        if grad_mask is not None:
          drift = drift * grad_mask

      # Normalize drift field
      boltzmann = (4 * psi + f * samples_x.pow(2).sum(dim=-1)) / (g**2)
      drift = (4 * drift + f * 2 * samples_x) / (g**2)
      drift_norm = torch.sqrt(torch.sum(drift**2, axis=-1, keepdims=True))
      drift = drift * math.sqrt(data_dim) / (drift_norm + 1e-6)

      # Compute MALA proposal step
      # step_size = step_size * 0.1
      diffusion = torch.randn_like(x) * dcoeff
      delta = weight * drift * step_size + diffusion * math.sqrt(2 * step_size)

      x = x + delta
      # samples_x = x_new
      # samples_x.requires_grad = True

      # with torch.enable_grad():
      #   net_fn = get_predict_fn(sde, model, train=False)
      #   samples_net = torch.cat([samples_x, cond_samples, std_enc], dim=-1)
      #   if sde.config.training.class_guidance:
      #     samples_net = torch.cat([samples_x, cond_samples, std_enc, labels], dim=-1)
      #   psi_new = net_fn(samples_net).squeeze(dim=-1)

      #   # Compute drift at x_new
      #   drift_new = torch.autograd.grad(psi_new, samples_x, torch.ones_like(psi_new))[0]
      #   drift_new = (4 * drift_new + f * 2 * samples_x) / (g**2)
      #   drift_norm = torch.sqrt(torch.sum(drift_new**2, axis=-1, keepdims=True))
      #   drift_new = drift_new * math.sqrt(data_dim) / (drift_norm + 1e-6)

      # # Compute log proposal probabilities
      # diff_x = x_new - x
      # log_q_x_to_xnew = -torch.sum((diff_x - step_size * drift) ** 2, dim=-1) / (8 * step_size)
      # log_q_xnew_to_x = -torch.sum((diff_x + step_size * drift_new) ** 2, dim=-1) / (8 * step_size)

      # # Compute MALA acceptance probability
      # log_alpha = psi - psi_new + log_q_xnew_to_x - log_q_x_to_xnew
      # log_alpha = torch.log_softmax(log_alpha, dim=-1)  
      # alpha = torch.exp(log_alpha)

      # # Generate uniform random samples for comparison
      # acceptance_mask = torch.rand_like(alpha) < alpha
      # # Apply element-wise acceptance
      # x = torch.where(acceptance_mask.unsqueeze(-1), x_new, x)

      # time.sleep(0.1)
      clear_output(wait=False)
      print(i, z)

      # print(t, state['t_eval'], step_size)
      # print(acceptance_mask.float().sum().item())
      print(psi.mean().item(), drift_norm.mean().item())

      Time.append(z)
      Norm.append(drift_norm.mean().item())
      # Energy.append(psi.mean().item())
      Energy.append(boltzmann.mean().item())

      if longrun and z >= time_eval[plot_idx]:
          X_samples.append(x.clone())
          plot_idx += 1

    if longrun:
      x = torch.stack(X_samples, dim=0)
      x = x.permute(1, 0, 2)

##================================================================================================================##
  else:
    def ode_func(z, x):

      # Prepare potential network input
      if mode == 'time':
        if z > ubound:
          t = ubound
        else:
          t = z
        # lamb = l_t(t)
      elif mode == 'lamb':
        if z > ubound:
          lamb = ubound
        else:
          lamb = z
      # t = t_l(lamb)

      if sde.config.training.augment_t:
        if sde.config.model.temb_type == 'lamb': 
          if sde.config.training.scheduling == 'cosine':
            lamb = 2 * math.log(math.tan(t*math.pi/2))
          elif sde.config.training.scheduling == 'fmot':
            lamb = 1 * math.log((t)**1/(1-t)**1)
          elif sde.config.training.scheduling == 'vapo':
            lamb = -1 * math.log((1-t)**2/(t)**1)
          elif sde.config.training.scheduling == 'elbo':
            lamb = -1 * math.log((1-t)**2/(2*t**2))
          if sde.config.training.shift_schedule:
            lamb += math.log(64 / sde.config.data.image_size)
      
      # state['t_eval'] += 1; step_size = abs(t) / state['t_eval']

      samples_x = from_flattened_numpy(x, (sample_size, -1)).to(device).type(torch.float32)
      # samples_x[:,:-1] += torch.randn_like(samples_x[:,:-1]) * math.sqrt(0.01)
      if sde.config.training.augment_z: 
        z = z_t(t)
        samples_x = torch.cat([samples_x[:,:-1], z * torch.ones(sample_size, 1).to(device).type(torch.float32)], dim=-1)
      samples_x.requires_grad = True

      with torch.enable_grad():
        # Get model function
        net_fn = get_predict_fn(sde, model, train=False)

        # Predict scalar potential (FC)
        # samples_net = samples_x
        if sde.config.training.augment_t:
          if sde.config.model.temb_type == 'time':
            cond = t
          elif sde.config.model.temb_type == 'lamb': 
            cond = lamb
          cond_samples = cond * torch.ones(sample_size, 1).to(device).type(torch.float32)
          std_enc = (1-t) * torch.ones(sample_size, 1).to(device).type(torch.float32)
          # temb = GaussianFourierProjection(sde.config.model.temb_dim, emb_type='fourier')(cond_samples)
          if sde.config.training.action_matching:
            samples_net = torch.cat([samples_x, cond_samples, std_enc], dim=-1)
          else:
            temb = get_timestep_embedding(cond_samples.squeeze(dim=-1), sde.config.model.temb_dim)
            samples_net = torch.cat([samples_x, temb], dim=-1)
        if sde.config.training.class_guidance:
          # 1-airplane  2-automobile  3-bird  4-cat  5-deer  6-dog  7-frog  8-horse  9-ship  10-truck
          # labels = torch.tensor([0,0,0,1,0,0,0,0,0,0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
          # samples_net = torch.cat([samples_x, cond_samples, labels], dim=-1)
          # labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
          labels = torch.zeros(sample_size, config.data.classes).to(device)
          samples_net = torch.cat([samples_x, cond_samples, std_enc, labels], dim=-1)
          
        psi = net_fn(samples_net)
        psi = psi.squeeze(dim=-1)
        # phi = phi.detach()
        # if sde.config.training.class_guidance:
        #   psi_null = net_fn(samples_net_null).squeeze(dim=-1)

        # Normalize field by its mean
        # psi -= psi.mean(dim=0, keepdim=True)

        # Compute (backpropagate) N-dimensional Poisson field (gradient)
        drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi))[0]
        # if sde.config.training.class_guidance:
        #   drift_null = torch.autograd.grad(psi_null, samples_x, torch.ones_like(psi))[0]
        #   w = sde.config.sampling.guidance_strength
        #   drift = (1 + w) * drift - w * drift_null

      # Predicted normalized Poisson field
      if mode == 'lamb':
        dt_dl = t*(1-t)
        drift = drift * dt_dl

      if grad_mask is not None:
        drift = drift * grad_mask
      
      # Normalize field
      drift_norm =  torch.sqrt(torch.sum(drift**2, axis=-1, keepdims=True))
      drift = drift * math.sqrt(data_dim) / (drift_norm + 1e-8)
      drift = weight * drift

      # time.sleep(0.1)
      clear_output(wait=False)
      print(z)

      # print(t, state['t_eval'], step_size)
      # if sde.config.training.augment_z:
      #   print(z, diff.item())
      # print(drift.shape)
      print(psi.mean().item(), drift_norm.mean().item())

      Time.append(z)
      Norm.append(drift_norm.mean().item())
      Energy.append(psi.mean().item())

      return to_flattened_numpy(drift)

    # Black-box ODE solver for the probability flow ODE.
    # Note that we use z = exp(t) for change-of-variable to accelearte the ODE simulation
    solution = integrate.solve_ivp(ode_func, boundary, to_flattened_numpy(x), rtol=rtol, atol=atol, method=method, t_eval=np.linspace(t_start, t_end, nplot))

    nfe = solution.nfev
    x = torch.tensor(solution.y[:,-1]).reshape(sample_size, -1)
    if longrun:
      x = torch.tensor(solution.y).T.reshape(nplot, sample_size, -1)
      x = x.permute(1, 0, 2)
    x = x.to(device).type(torch.float32)
    y = x
    if sde.config.training.augment_z: x, _ = torch.split(x, [data_dim, 1], dim=-1)
##================================================================================================================##
  
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Elapsed time: {elapsed_time:.6f} seconds")

  # Detach augmented z dimension
  x = x.reshape(-1, *new_shape[1:])
  x = inverse_scaler(x)
  # phi = phi.detach().reshape(new_shape)
  # phi = inverse_scaler(phi.detach().reshape(new_shape))
  n = nfe
  print(n)

sampling_plot_path = "/home/admin01/Junn/VAPO/results/sampling.png"
plt.figure(figsize=(20, 7.5))
plt.plot(Time, Energy, linestyle="-", linewidth=2, color="blue", label="Energy")
plt.plot(Time, Norm, linestyle="-", linewidth=2, color="red", label="Norm")

plt.xlabel("Iteration", fontsize=20)
# plt.ylabel(r"$\mathbf{E}[\|\nabla_{\!x} \Phi\|^2]$", fontsize=20)
# plt.ylabel(r"$\mathbf{E}[\|\Phi\|^2]$", fontsize=20)
plt.title("SDE (Langevin Dynamics) Sampling", fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.xlim([plt.gca().get_xlim()[0], 300000])
# plt.ylim([plt.gca().get_ylim()[0], 0.001])
plt.grid(True)
plt.legend([r"$\mathbf{E}[\|\Phi\|^2]$", r"$\mathbf{E}[\|\nabla_{\!x} \Phi\|^2]$"], fontsize=20, loc='best')
plt.savefig(sampling_plot_path)
plt.show()

# sample = inverse_scaler(x_prior.view(shape))
# nrow = int(np.sqrt(sample.shape[0]))
# image_grid = make_grid(sample, nrow, padding=2)
# sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
# save_path = './sample_prior.png'
# with tf.io.gfile.GFile(save_path, "wb") as fout:
#   save_image(image_grid, fout)
# img = plt.imread(save_path) 
# plt.imshow(img)
# plt.show()

if batch is not None:
  sample = inverse_scaler(batch.view(shape))
  nrow = int(np.sqrt(sample.shape[0]))
  image_grid = make_grid(sample, nrow, padding=2)
  sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
  save_path = './sample_data.png'
  with tf.io.gfile.GFile(save_path, "wb") as fout:
    save_image(image_grid, fout)
  img = plt.imread(save_path) 
  plt.imshow(img)
  plt.show()

# sample = phi.clone()
# if inpainting:
#   sample = torch.where(mask, sample, inverse_scaler(batch))
# nrow = int(np.sqrt(sample.shape[0]))
# image_grid = make_grid(sample, nrow, padding=2)
# sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
# # save_path = './sample_x.png'
# with tf.io.gfile.GFile(save_path_sample, "wb") as fout:
#   save_image(image_grid, fout)
# img = plt.imread(save_path_sample) 
# plt.imshow(img)
# plt.show()

sample = x.clone()
if inpainting:
  sample = torch.where(mask, sample, inverse_scaler(batch))
nrow = int(np.sqrt(sample.shape[0]))
if longrun:
  nrow = nplot
image_grid = make_grid(sample, nrow, padding=2)
sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
# save_path = './sample_x.png'
with tf.io.gfile.GFile(save_path_sample, "wb") as fout:
  save_image(image_grid, fout)
img = plt.imread(save_path_sample) 
plt.imshow(img)
plt.show()

ema.restore(model.parameters())

# %% plot.py
import matplotlib.pyplot as plt
import pandas as pd
import re

# Define file path
file_path = "/home/admin01/Junn/VAPO/homotopy_planar/checkpoints/cifar10_stdout.txt"

# Read file contents
with open(file_path, "r") as file:
    lines = file.readlines()

# Regular expression to extract step, norm, and nll values
pattern = re.compile(r"step: (\d+), .* norm: ([\de\+\-\.]+), nll: ([\de\+\-\.]+)")

# Extract data
data = []
for line in lines:
    match = pattern.search(line)
    if match:
        iter = int(match.group(1))
        norm = 2 * float(match.group(2)) * 3072
        nll = float(match.group(3))
        data.append([iter, norm, nll, norm/nll])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Iter", "Norm", "NLL", "Poincare"])

# Save to CSV
csv_path = "/home/admin01/Junn/VAPO/results/plot.csv"
df.to_csv(csv_path, index=False)

poincare_plot_path = "/home/admin01/Junn/VAPO/results/poincare.png"
plt.figure(figsize=(20, 7.5))
# plt.plot(df["Iter"], df["Norm"], label="", linestyle="-", linewidth=2, color="blue")
# plt.plot(df["Iter"], df["NLL"], label="", linestyle="-", linewidth=2, color="blue")
plt.plot(df["Iter"], df["Poincare"], label="", linestyle="-", linewidth=2, color="blue")

min_poincare = df["Poincare"].min()
plt.axhline(y=min_poincare, color="red", linestyle="--", linewidth=3, label=f"Poincaré constant = {min_poincare:e}")

plt.xlabel("Iteration", fontsize=20)
plt.ylabel(r"$\frac{\mathbf{E}[\|\nabla_{\!x} \Phi\|^2]}{\mathbf{E}[\|\Phi\|^2]}$", fontsize=20)
# plt.title("Poincaré constant", fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.xlim([plt.gca().get_xlim()[0], 300000])
# plt.ylim([plt.gca().get_ylim()[0], 0.001])
plt.grid(True)
plt.legend(fontsize=20)
plt.savefig(poincare_plot_path)
plt.show()