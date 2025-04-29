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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from scipy import integrate
from models import utils as mutils
from methods import VESDE, VPSDE
from models import utils_poisson
import datasets
import lamb
import logging
# from models.ebm_models import get_timestep_embedding
# from models.ebm_models import GaussianFourierProjection
# from sklearn.preprocessing import OneHotEncoder
from einops import rearrange


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    beta1 = config.optim.beta1
    if beta1 == 0:
       beta2 = 0.9
    else:
       beta2 = 0.999
       
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(beta1, beta2), eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(beta1, beta2), eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'Lamb':
        optimizer = lamb.Lamb(params, lr=config.optim.lr, betas=(beta1, beta2), eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.optim.lr, momentum=0.9, weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer

def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip,
                    grad_clip_mode=config.optim.grad_clip_mode,
                    anneal_rate=config.optim.anneal_rate,
                    anneal_iters=config.optim.anneal_iters):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if step > warmup:
          # if step in np.array(anneal_epochs) * math.ceil(config.data.size / config.training.batch_size):
          if step in anneal_iters:
              for g in optimizer.param_groups:
                  new_lr = g['lr'] * anneal_rate
                  g['lr'] = new_lr
              logging.info("Decaying lr to {}".format(new_lr))
        if grad_clip >= 0:
            if grad_clip_mode == 'norm':
              torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            elif grad_clip_mode == 'std':
              clip_grad(optimizer, grad_clip)
        optimizer.step()

    return optimize_fn

def clip_grad(optimizer, grad_clip):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'step' not in state or state['step'] < 1:
                    continue
                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']
                bound = grad_clip * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def get_perturb_batch_loss_fn(sde, train, reduce_mean=True, continuous=True, eps=1e-8, method_name=None, optimize_fn=None, sampling_fn=None):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `methods.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `Truec` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """

    def loss_fn(model, batch, state, train, labels=None, sample_bool=False, ind_bool=True):
        """Compute the loss function.

        Args:
          model: A PFGM or score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """

        # perturb sigmas
        if method_name == 'homotopy':

          # Get configs
          data_dim = sde.config.data.dim
          batch_size = sde.config.training.batch_size
          ensemble_size = sde.config.training.small_batch_size
          if not ind_bool:
            assert batch_size == ensemble_size
          # num_particles = sde.config.training.num_particles
          sample_size = sde.config.training.sample_size

          # Get the mini-batch
          batch = batch.reshape(batch_size, -1)
          samples_batch = batch[:ensemble_size, :data_dim]
          device = batch.device

          # Get one-hot label encodings
          if sde.config.training.class_guidance:
            # labels_nested = [[label] for label in labels]m
            # encoder = OneHotEncoder(categories='auto', sparse_output=False)
            # labels = encoder.fit_transform(labels_nested)
            # labels = torch.from_numpy(labels).to(device)
            labels = F.one_hot(labels, num_classes=sde.config.data.classes)
            # logging.info(labels.shape)
            # ----labels shape torch.Size([batch_size, num of labels])
            # Random 
            random = torch.rand(labels.shape[0]).to(device)
            # Should probably turn this into an arg
            p_uncond = sde.config.training.p_uncond
            uncond_mask = (random < p_uncond)[:,None]
            # What should null be -> [0,0,0,0,0,0,0,0,0,0]? 
            null = torch.zeros(labels.shape[-1]).to(device)
            labels =  torch.where(uncond_mask, null, labels)
            # labels = labels_null

          # Get prior sigma
          # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels)
          # sigma_prior = math.sqrt(sde.config.data.image_size) * sde.config.data.channels
          # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.channels)
          mean_prior = sde.config.training.mean_prior
          sigma_prior = sde.config.training.sigma_prior

          # Change-of-variable: z = -ln(t)
          z_t = lambda t: -torch.log(t)
          t_z = lambda z: torch.exp(-z)

          with torch.no_grad():
            
            # Sample t from log-uniform distribution
            t_range = (sde.config.training.t_start, sde.config.training.t_end)
            t_max = sde.config.training.t_max

            # Noise scheduling
            if sde.config.training.continuous:
              t_samples = t_range[0] + torch.rand(ensemble_size).to(device) * (t_range[1] - t_range[0])
            else:
              t_samples = t_range[0] + (torch.randint(0, 1000, (ensemble_size, 1)).to(device) / 1000) * (t_range[1] - t_range[0]) 
            t_clipped = t_samples.clone()
            t_clipped[t_samples > t_max] = t_max
            t_samples = t_samples[:,None]
            t_clipped = t_clipped[:,None]

            mean_dt = 1 * torch.ones_like(t_clipped)
            std_dt = -1 * torch.ones_like(t_clipped)
            mean_enc_batch = t_clipped
            var_enc = (1-t_clipped)

            # Compute enc (posterior) mean and var
            # sigma_dec = sde.config.training.sigma_min
            # var_dec = sigma_dec**2
            # tao_samples = torch.exp(lamb_samples) * var_dec
            # var_prior = sigma_prior**2
            # mean_enc_batch = (tao_samples * var_prior) / (var_dec + tao_samples * var_prior)
            # # mean_enc_prior = var_dec / (var_dec + tao_samples * var_prior)
            # var_enc = var_prior * var_dec / (var_dec + tao_samples * var_prior)

            # Perturb data samples with gaussians
            gaussians_x = torch.randn(ensemble_size, data_dim).to(device)
            mean_enc = mean_enc_batch.pow(sde.config.training.mean_power) * samples_batch
            std_enc = var_enc.pow(sde.config.training.var_power)
            samples_x = mean_enc + (std_enc + sde.config.training.sigma_eps) * gaussians_x
            const = data_dim * var_enc + (1 - mean_enc_batch).pow(2) * samples_batch.pow(2).mean(dim=-1, keepdim=True)

            if sample_bool:
              samples_s, nfe = sampling_fn(model, state, sample_size=sample_size, method='RK23', eps=1e-3, rtol=1e-3, atol=1e-3, inverse_scale=False)
              print("step: %d, nfe: %d" % (state['step'], nfe))
              # print(samples_x.min(), samples_x.max(), samples_s.min(), samples_s.max())
              samples_x[:sample_size] = samples_s

          with torch.enable_grad():
            # Get model function
            net_fn = mutils.get_predict_fn(sde, model, train=train, continuous=continuous)

            # Predict scalar potential
            samples_x.requires_grad = True
            # time embeddings
            if sde.config.training.augment_t:
              if sde.config.model.temb_type == 'time':
                cond_samples = t_samples
              elif sde.config.model.temb_type == 'lamb': 
                cond_samples = 1 * torch.log((t_clipped) / (1-t_clipped))
              cond_samples.requires_grad = True
              if sde.config.training.class_guidance:
                samples_net = torch.cat([samples_x, cond_samples, std_enc, labels], dim=-1)
              else:
                samples_net = torch.cat([samples_x, cond_samples, std_enc], dim=-1)
            else:
              samples_net = samples_x
            psi = net_fn(samples_net).squeeze(dim=-1)
            Reg = psi.pow(2).mean()

            # Normalize potential by its mean
            # psi -= psi.mean(dim=0, keepdim=True)

            # Compute (backpropagate) N-dimensional Poisson field (gradient)
            if sde.config.training.augment_t:
              drift_x, drift_emb = torch.autograd.grad(psi, [samples_x, cond_samples], torch.ones_like(psi), create_graph=True)
            else:
              drift_x = torch.autograd.grad(psi, samples_x, torch.ones_like(psi), create_graph=True)[0]
            # laplacian_x = torch.autograd.grad(drift_x, samples_x, torch.ones_like(drift_x), create_graph=True)[0]

          # Compute drift norm
          if sde.config.training.augment_t: 
            Norm_emb = drift_emb.pow(2) * (1-t_clipped).pow(sde.config.training.std_power_temb)
            Norm_emb = Norm_emb.sum(dim=-1).mean()
          if sde.config.training.reduce_mean:
            Norm_emb = Norm_emb.mean()
          else:
            Norm_emb = Norm_emb.sum(dim=-1).mean()

          if sde.config.training.entropic:
            etp_cond = - gaussians_x * (sde.config.training.coeff_diffusion)**2
            Norm_etp = (drift_x - etp_cond).pow(2) * (1-t_clipped).pow(sde.config.training.std_power_norm)
          else:
            Norm_etp = drift_x.pow(2) * (1-t_clipped).pow(sde.config.training.std_power_norm)
          if sde.config.training.reduce_mean:
            Norm_etp = Norm_etp.mean()
          else:
            Norm_etp = Norm_etp.sum(dim=-1).mean()

          if sde.config.optim.laplacian_mode == 'score':
            vf_cond = (std_dt * gaussians_x)
          elif sde.config.optim.laplacian_mode == 'fmot':
            vf_cond = (mean_dt * batch) + (std_dt * gaussians_x)
            
          if sde.config.optim.cvf_mode == 'dist':
            Laplacian = (drift_x - vf_cond).pow(2) * (1-t_clipped).pow(sde.config.training.std_power_lap)
          elif sde.config.optim.cvf_mode == 'dot':
            Laplacian = - (drift_x * vf_cond) * (1-t_clipped).pow(sde.config.training.std_power_lap)
          elif sde.config.optim.cvf_mode == 'cos':
            Laplacian = - (drift_x * vf_cond)
            Norm_lap = (drift_x.norm(dim=-1) * vf_cond.norm(dim=-1))[:, None] + 1e-8
            Laplacian = Laplacian / Norm_lap.pow(1 - (1-t_clipped).pow(sde.config.training.std_power_lap))
            # Laplacian = - F.cosine_similarity(drift_x, vf_cond, dim=-1)[:, None] * (1-t_clipped).pow(sde.config.training.std_power_lap)
          if sde.config.training.reduce_mean:
            Laplacian = Laplacian.mean()
          else:
            Laplacian = Laplacian.sum(dim=-1).mean()
          
          # Vel = - drift_emb * (1-t_clipped).pow(sde.config.training.std_power_vel)
          # Vel = Vel.mean()
            
          if sde.config.training.method == 'posterior':  

            with torch.no_grad():
              # Compute Normalized Innovation Squared (Gamma)
              innovation = - std_dt * gaussians_x.pow(2) - mean_dt * (batch * gaussians_x)
              innovation = innovation * (1-t_clipped).pow(sde.config.training.std_power_cov)
              if sde.config.training.reduce_mean:
                Gamma = innovation.mean(dim=-1)
              else:
                Gamma = innovation.sum(dim=-1)
              Gamma = Gamma - Gamma.mean()

          # Compute sample correlation between potential and NIS
          Cov = (Gamma * psi).mean()
          if sde.config.optim.cov_mode == 'cov':
            Corr = Cov / (sde.config.training.divisor + eps)
          elif sde.config.optim.cov_mode == 'corr':
            Corr = Cov / (Gamma.std() * psi.detach().std() + eps)
            
          Loss = torch.zeros_like(Corr)
          Loss += sde.config.training.weight_cov * Corr
          
          # Loss += sde.config.training.weight_vel * Vel

          Norm = sde.config.training.weight_norm * Norm_etp
          if sde.config.training.augment_t:
            Norm += sde.config.training.weight_temb * Norm_emb
          Loss += Norm
          
          Loss += sde.config.training.weight_lap * Laplacian
                                
          return Loss, Corr, Laplacian, Norm, Reg
        
    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, sampling_fn=None, reduce_mean=False, method_name=None):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `methods.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.

    Returns:
      A one-step function for training or evaluation.
    """

    perturb_loss_fn = get_perturb_batch_loss_fn(sde, train, reduce_mean=reduce_mean, continuous=True, method_name=method_name, optimize_fn=optimize_fn, sampling_fn=sampling_fn)

    def step_fn(state, batch, labels=None, sample_bool=False, ind_bool=False):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the PFGM or score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            # Sample bool
            if sde.config.training.sample_freq > 0 and state['step'] % sde.config.training.sample_freq == 0 and state['step'] > sde.config.optim.warmup: 
              sample_bool = True
            Loss, Corr, Laplacian, Norm, Reg = perturb_loss_fn(model, batch, state, train, labels=labels, sample_bool=sample_bool, ind_bool=ind_bool)
            optimizer = state['optimizer']
            optimizer.zero_grad()
            Loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            # if state['sigma_max'] > sde.config.training.sigma_clip: 
            #    state['sigma_max'] *= 1 - sde.config.training.sigma_anneal
            # else: 
            #    state['sigma_max'] = sde.config.training.sigma_clip
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                Loss, Corr, Laplacian, Norm, Reg = perturb_loss_fn(model, batch, state, train, labels=labels)

        return Loss, Corr, Laplacian, Norm, Reg

    return step_fn
