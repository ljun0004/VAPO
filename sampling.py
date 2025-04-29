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
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc
import math
import pickle
import logging

from models.utils import from_flattened_numpy, to_flattened_numpy, get_predict_fn
from scipy import integrate
import methods
from models import utils as mutils
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import datasets
# from IPython.display import clear_output
# from models.ebm_models import get_timestep_embedding
# from models.ebm_models import GaussianFourierProjection

_CORRECTORS = {}
_PREDICTORS = {}
_ODESOLVER = {}

def register_odesolver(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _ODESOLVER:
      raise ValueError(f'Already registered model with name: {local_name}')
    _ODESOLVER[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)

def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]

def get_ode_solver(name):
  return _ODESOLVER[name]

def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps=1e-10, rtol=1e-4, atol=1e-4):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `methods.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    if config.sampling.ode_solver == 'rk45':
      if config.training.sde == 'homotopy':
        # RK45 ode sampler for PATH
        sampling_fn = get_rk45_sampler_path(sde=sde,
                                           shape=shape,
                                           inverse_scaler=inverse_scaler,
                                           eps=eps, rtol=rtol, atol=atol,
                                           device=config.device)
      elif config.training.sde == 'poisson':
        # RK45 ode sampler for PFGM
        sampling_fn = get_rk45_sampler_pfgm(sde=sde,
                                           shape=shape,
                                           inverse_scaler=inverse_scaler,
                                           eps=eps, rtol=rtol, atol=atol,
                                           device=config.device)
      else:
        sampling_fn = get_rk45_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps, rtol=rtol, atol=atol,
                                      device=config.device)
    else:
      ode_solver = get_ode_solver(config.sampling.ode_solver.lower())
      sampling_fn = get_ode_sampler(sde=sde,
                                    shape=shape,
                                    ode_solver=ode_solver,
                                    inverse_scaler=inverse_scaler,
                                    eps=eps, rtol=rtol, atol=atol,
                                    device=config.device)

  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps, rtol=rtol, atol=atol,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn

class ODE_Solver(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, net_fn, eps=None):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    if sde.config.training.sde not in ['homotopy','poisson']:
      self.rsde = sde.reverse(net_fn, probability_flow=True)
    self.net_fn = net_fn
    self.eps = eps

  @abc.abstractmethod
  def update_fn(self, x, t, t_list=None, idx=None):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, net_fn, probability_flow=False, eps=None):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    if sde.config.training.sde not in ['homotopy','poisson']:
      self.rsde = sde.reverse(net_fn, probability_flow)
    self.net_fn = net_fn
    self.eps = eps

  @abc.abstractmethod
  def update_fn(self, x, t, t_list=None, idx=None):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, net_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.net_fn = net_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, net_fn, probability_flow=False, eps=None):
    super().__init__(sde, net_fn, probability_flow, eps)

  def update_fn(self, x, t, t_list=None, idx=None):
    z = torch.randn_like(x)
    if self.sde.config.training.sde in ['homotopy','poisson']:
      if t_list is None:
        dt = - (np.log(self.sde.config.sampling.z_max) - np.log(self.eps)) / self.sde.N
      else:
        # integration over z
        dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
        dt = float(dt.cpu().numpy())
      drift = self.sde.ode(self.net_fn, x, t)
      diffusion = torch.zeros((len(x)), device=x.device)
    else:
      if t_list is None:
        dt = -1. / self.sde.N
      drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean

@register_odesolver(name='forward_euler')
class ForwardEulerPredictor(ODE_Solver):
  def __init__(self, sde, net_fn, eps=None):
    super().__init__(sde, net_fn, eps)

  def update_fn(self, x, t, t_list=None, idx=None):

    if self.sde.config.training.sde in ['homotopy','poisson']:
      # dt = - (np.log(self.sde.config.sampling.z_max) - np.log(self.eps)) / self.sde.N
      drift = self.sde.ode(self.net_fn, x, t)
      if t_list is None:
        dt = - (np.log(self.sde.config.sampling.z_max) - np.log(self.eps)) / self.sde.N
      else:
        # integration over z
        dt = - (1 - torch.exp(t_list[idx + 1] - t_list[idx]))
        dt = float(dt.cpu().numpy())
    else:
      dt = -1. / self.sde.N
      drift, _ = self.rsde.sde(x, t)
    x = x + drift * dt
    return x

@register_odesolver(name='improved_euler')
class ImprovedEulerPredictor(ODE_Solver):
  def __init__(self, sde, net_fn, eps=None):
    super().__init__(sde, net_fn, eps)

  def update_fn(self, x, t, t_list=None, idx=None):
    if self.sde.config.training.sde in ['homotopy','poisson']:
      if t_list is None:
        dt = - (np.log(self.sde.config.sampling.z_max) - np.log(self.eps)) / self.sde.N
      else:
        # integration over z
        dt = (torch.exp(t_list[idx + 1] - t_list[idx]) - 1)
        dt = float(dt.cpu().numpy())
      drift = self.sde.ode(self.net_fn, x, t)
    else:
      dt = -1. / self.sde.N
      drift, _ = self.rsde.sde(x, t)
    x_new = x + drift * dt

    if idx == self.sde.N - 1:
      return x_new
    else:
      idx_new = idx + 1
      t_new = t_list[idx_new]
      t_new = torch.ones(len(t), device=t.device) * t_new

      if self.sde.config.training.sde in ['homotopy','poisson']:
        if t_list is None:
          dt_new = - (np.log(self.sde.config.sampling.z_max) - np.log(self.eps)) / self.sde.N
        else:
          # integration over z
          dt_new = (1 - torch.exp(t_list[idx] - t_list[idx+1]))
          dt_new = float(dt_new.cpu().numpy())
        drift_new = self.sde.ode(self.net_fn, x_new, t_new)
      else:
        drift_new, diffusion = self.rsde.sde(x_new, t_new)
        dt_new = -1. / self.sde.N

      x = x + (0.5 * drift * dt + 0.5 * drift_new * dt_new)
      return x



@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, net_fn, probability_flow=False, eps=None):
    super().__init__(sde, net_fn, probability_flow, eps)

  def update_fn(self, x, t, t_list=None, idx=None):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, net_fn, probability_flow=False):
    pass

  def update_fn(self, x, t, t_list=None, idx=None):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, net_fn, snr, n_steps):
    super().__init__(sde, net_fn, snr, n_steps)
    if not isinstance(sde, methods.VPSDE) \
        and not isinstance(sde, methods.VESDE) \
        and not isinstance(sde, methods.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    net_fn = self.net_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, methods.VPSDE) or isinstance(sde, methods.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = net_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, net_fn, snr, n_steps):
    super().__init__(sde, net_fn, snr, n_steps)
    if not isinstance(sde, methods.VPSDE) \
        and not isinstance(sde, methods.VESDE) \
        and not isinstance(sde, methods.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    net_fn = self.net_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, methods.VPSDE) or isinstance(sde, methods.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = net_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, net_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x

def shared_ode_solver_update_fn(x, t, sde, model, ode_solver, eps, t_list=None, idx=None):
  """A wrapper that configures and returns the update function of ODE solvers."""
  net_fn = mutils.get_predict_fn(sde, model, train=False, continuous=True)
  ode_solver_obj = ode_solver(sde, net_fn, eps)
  return ode_solver_obj.update_fn(x, t, t_list=t_list, idx=idx)

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous, eps, t_list=None, idx=None):
  """A wrapper that configures and returns the update function of predictors."""
  net_fn = mutils.get_predict_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, net_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, net_fn, probability_flow, eps)
  return predictor_obj.update_fn(x, t, t_list=t_list, idx=idx)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  net_fn = mutils.get_predict_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, net_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, net_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `methods.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          eps=eps)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model):
    """ The PC sampler funciton.

    Args:
      model: A PFGM or score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device).float()
      if sde.config.training.sde in ['homotopy','poisson']:
        timesteps = torch.linspace(np.log(sde.config.sampling.z_max), np.log(eps), sde.N + 1, device=device).float()
      else:
        timesteps = torch.linspace(sde.T, eps, sde.N+1, device=device).float()

      for i in tqdm(range(sde.N)):
        t = timesteps[i]
        if sde.config.training.sde in ['homotopy','poisson']:
          vec_t = torch.ones(shape[0], device=t.device).float() * t
          x, x_mean = corrector_update_fn(x, vec_t, model=model)
          x, x_mean = predictor_update_fn(x, vec_t, model=model, t_list=timesteps, idx=i)
        else:
          vec_t = torch.ones(shape[0], device=t.device).float() * t
          x, x_mean = corrector_update_fn(x, vec_t, model=model)
          x, x_mean = predictor_update_fn(x, vec_t, model=model)


      return inverse_scaler(x_mean if denoise else x), sde.N if sde.config.sampling.corrector == 'none' else sde.N * (n_steps + 1)

  return pc_sampler

def get_ode_sampler(sde, shape, ode_solver, inverse_scaler, eps=1e-3, device='cuda'):
  """Create a ODE sampler, for foward Euler or Improved Euler method.

  Args:
    sde: An `methods.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    ode_solver: A subclass of `sampling.ODE_Solver` representing the predictor algorithm.
    inverse_scaler: The inverse data normalizer.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  ode_update_fn = functools.partial(shared_ode_solver_update_fn,
                                    sde=sde,
                                    ode_solver=ode_solver,
                                    eps=eps)

  def ode_sampler(model):
    """ The ODE sampler funciton.

    Args:
      model: A PFGM or score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device).float()
      if sde.config.training.sde in ['homotopy','poisson']:
        timesteps = torch.linspace(np.log(sde.config.sampling.z_max), np.log(eps), sde.N + 1, device=device).float()
      else:
        timesteps = torch.linspace(sde.T, eps, sde.N+1, device=device).float()

      imgs = []
      for i in tqdm(range(sde.N)):
        t = timesteps[i]
        if sde.config.training.sde in ['homotopy','poisson']:
          vec_t = torch.ones(shape[0], device=t.device).float() * t
          x = ode_update_fn(x, vec_t, model=model, t_list=timesteps, idx=i)
        else:
          vec_t = torch.ones(shape[0], device=t.device).float() * t
          x = ode_update_fn(x, vec_t, model=model)

      #   image_grid = make_grid(inverse_scaler(x), nrow=int(np.sqrt(len(x))))
      #   im = Image.fromarray(
      #     image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
      #   imgs.append(im)
      #
      # import os
      # imgs[0].save(os.path.join("celeba_movie_50.gif"), save_all=True, append_images=imgs[1:],
      #              duration=1, loop=0)
      # exit(0)
      return inverse_scaler(x), 2 * sde.N - 1 if sde.config.sampling.ode_solver == 'improved_euler' else sde.N

  return ode_sampler



def get_rk45_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-4, atol=1e-4,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `methods.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    net_fn = get_predict_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, net_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x


  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    net_fn = get_predict_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(net_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler


def get_rk45_sampler_pfgm(sde, shape, inverse_scaler, rtol=1e-4, atol=1e-4,
                    method='RK45', eps=1e-3, device='cuda'):

  """RK45 ODE sampler for PFGM.

  Args:
    sde: An `methods.SDE` object that represents PFGM.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def ode_sampler(model, state, x=None):

    with torch.no_grad():
      # Initial sample
      if x is None:
        x = sde.prior_sampling(shape).to(device)

      z = torch.ones((len(x), 1, 1, 1)).to(x.device)
      z = z.repeat((1, 1, sde.config.data.image_size, sde.config.data.image_size)) * sde.config.sampling.z_max
      x = x.view(shape)
      # Augment the samples with extra dimension z
      # We concatenate the extra dimension z as an addition channel to accomondate this solver
      x = torch.cat((x, z), dim=1)
      x = x.float()
      new_shape = (len(x), sde.config.data.channels + 1, sde.config.data.image_size, sde.config.data.image_size)

      def ode_func(t, x):

        if sde.config.sampling.vs:
          print(np.exp(t))
        x = from_flattened_numpy(x, new_shape).to(device).type(torch.float32)
        # Change-of-variable z=exp(t)
        z = np.exp(t)

        # Get score-based network function
        net_fn = get_predict_fn(sde, model, train=False)

        x_drift, z_drift = net_fn(x[:, :-1], torch.ones((len(x))).cuda() * z)
        x_drift = x_drift.view(len(x_drift), -1)

        # Substitute the predicted z with the ground-truth
        # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
        z_exp = sde.config.sampling.z_exp
        if z < z_exp and sde.config.training.gamma > 0:
          data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
          sqrt_dim = np.sqrt(data_dim)
          norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
          x_norm = sde.config.training.gamma * norm_1 / (1 - norm_1)
          x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
          z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + sde.config.training.gamma)

        # Predicted normalized Poisson field
        v = torch.cat([x_drift, z_drift[:, None]], dim=1)
        dt_dz = 1 / (v[:, -1] + 1e-5)
        dx_dt = v[:, :-1].view(shape)

        # Get dx/dz
        dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
        # drift = z * (dx/dz, dz/dz) = z * (dx/dz, 1)
        drift = torch.cat([z * dx_dz,
                           torch.ones((len(dx_dz), 1, sde.config.data.image_size,
                                       sde.config.data.image_size)).to(dx_dz.device) * z], dim=1)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE.
      # Note that we use z = exp(t) for change-of-variable to accelearte the ODE simulation
      solution = integrate.solve_ivp(ode_func, (np.log(sde.config.sampling.z_max), np.log(eps)), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)

      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(new_shape).to(device).type(torch.float32)

      # Detach augmented z dimension
      x = x[:, :-1]
      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler

def get_rk45_sampler_path(sde, shape, inverse_scaler, eps=1e-10, rtol=1e-4, atol=1e-4,
                    method='RK45', device='cuda'):

  """RK45 ODE sampler for PATH.

  Args:
    sde: An `methods.SDE` object that represents PATH.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  resize_op = transforms.Resize([sde.config.data.image_size_ori, sde.config.data.image_size_ori], antialias=True)

  def ode_sampler(model, state, x=None, mask=None, grad_mask=None, sample_size=shape[0], method=method, eps=eps, rtol=rtol, atol=atol, inverse_scale=True):
    # x: initial image samples of shape (B,C,H,W)
    # grad_mask: zero mask for drift of shape (B,C,H,W)
    # sample_size: batch size
    # print("in oder sampler")

    with torch.no_grad():
      data_dim = sde.config.data.channels * sde.config.data.image_size * sde.config.data.image_size
      # batch_size = sde.config.training.batch_size
      # ensemble_size = sde.config.training.small_batch_size
      # num_particles = sde.config.training.num_particles
      t_start = sde.config.training.t_start
      t_max = sde.config.training.t_max
      t_end = sde.config.sampling.t_end
      # var_dec = sde.config.training.sigma_min**2

      print("eps: %.5e, rtol: %.5e, atol: %.5e, t_start: %f, t_max: %f, t_end: %f, weight: %f, temb: %s, solver: %s" % (eps, rtol, atol, t_start, t_end, t_end, sde.config.sampling.weight, sde.config.model.temb_type, sde.config.sampling.solver))

      # Change-of-variable: z = -ln(t)
      z_t = lambda t: -math.log(t)
      t_z = lambda z: math.exp(-z)
    
      # Geometric sequence of sigmas
      # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels)
      # sigma_prior = math.sqrt(sde.config.data.image_size) * sde.config.data.channels
      # sigma_prior = math.sqrt(sde.config.data.image_size * sde.config.data.channels)
      mean_prior = sde.config.training.mean_prior
      sigma_prior = sde.config.training.sigma_prior
      # Sample from prior
      # torch.manual_seed(0)
      gaussian = torch.randn(sample_size, data_dim)
      x_init = mean_prior + sigma_prior * gaussian.to(device)

      new_shape = (sample_size, sde.config.data.channels, sde.config.data.image_size, sde.config.data.image_size)

      # Initial sample
      if x is None:
        x = x_init
        if sde.config.training.augment_z: 
          x = torch.cat([x, z_t(eps) * torch.ones(sample_size, 1).to(device)], dim=-1)
      else:
        if mask is not None:
          # x += x_init.reshape(new_shape) * mask
          x = torch.where(mask, x_init.reshape(new_shape), x)
        x = x.reshape(len(x), -1).to(device) 
        # sample_size = len(x)

      if grad_mask is not None:
        grad_mask = grad_mask.reshape(len(grad_mask), -1).to(device) 
        assert list(grad_mask.shape) == list(x.shape), f"{grad_mask.shape} != {x.shape}"

      # x = x.view(shape).float()
      # t = np.log(sde.config.sampling.z_max)
      # x = to_flattened_numpy(x)
      state['t_eval'] = 0

      if sde.config.sampling.solver == 'Euler':
        
        nfe = sde.config.sampling.nfe
        for t in np.linspace(t_start, t_end, nfe + 1):

          step_size =  abs(t - state['t_eval'])
          state['t_eval'] = t

          if t > t_max:
            tao = t_max
          else:
            tao = t
          
          state['t_eval'] += 1; step_size = abs(t) / state['t_eval']

          samples_x = x
          samples_x.requires_grad = True

          with torch.enable_grad():
            # Get model function
            net_fn = get_predict_fn(sde, model, train=False)

            # Predict scalar potential (FC)
            samples_net = samples_x
            if sde.config.training.augment_t:
              if sde.config.model.temb_type == 'time':
                cond = t
              elif sde.config.model.temb_type == 'lamb': 
                if sde.config.training.scheduling == 'cosine':
                  lamb = 2 * math.log(math.tan(tao*math.pi/2))
                elif sde.config.training.scheduling == 'fmot':
                  lamb = 1 * math.log((tao)/(1-tao))
                elif sde.config.training.scheduling == 'vapo':
                  lamb = 2 * math.log((tao)/(1-tao))
                elif sde.config.training.scheduling == 'elbo':
                  lamb = 1 * math.log((2*tao**2)/(1-tao)**2)
                if sde.config.training.shift_schedule:
                  lamb += math.log(sde.config.training.shift_ref / sde.config.data.image_size)
                cond = lamb
              cond_samples = cond * torch.ones(sample_size, 1).to(device).type(torch.float32)
              std_enc = (1-tao) * torch.ones(sample_size, 1).to(device).type(torch.float32)
              # temb = GaussianFourierProjection(sde.config.model.temb_dim, emb_type='fourier')(cond_samples)
              samples_net = torch.cat([samples_x, cond_samples, std_enc], dim=-1)
            if sde.config.training.class_guidance:
              # 1-airplane  2-automobile  3-bird  4-cat  5-deer  6-dog  7-frog  8-horse  9-ship  10-truck
              labels = torch.tensor([0,0,0,1,0,0,0,0,0,0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
              samples_net = torch.cat([samples_x, cond_samples, labels], dim=-1)
              labels_null = torch.tensor([0,0,0,0,0,0,0,0,0,0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
              samples_net_null = torch.cat([samples_x, cond_samples, labels_null], dim=-1)
              
            psi = net_fn(samples_net).squeeze(dim=-1)
            if sde.config.training.class_guidance:
              psi_null = net_fn(samples_net_null).squeeze(dim=-1)

            # Normalize field by its mean
            # psi -= psi.mean(dim=0, keepdim=True)

            # Compute (backpropagate) N-dimensional Poisson field (gradient)
            # print(sde.config.sampling.weight)
            drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi))[0]
            if sde.config.training.class_guidance:
              drift_null = torch.autograd.grad(psi_null, samples_x, torch.ones_like(psi))[0]
              w = sde.config.sampling.guidance_strength
              drift = (1 + w) * drift - w * drift_null

            if sde.config.model.temb_type == 'lamb':
              drift = drift * t*(1-t)

          # Normalize field
          # drift_norm =  torch.sqrt(torch.sum(drift**2, axis=-1, keepdims=True))
          # drift = drift / drift_norm
          # drift = drift * math.sqrt(data_dim)
          drift = drift * sde.config.sampling.weight
          
          # noise = torch.randn_like(x)
          # diffusion = math.sqrt(2*step_size) * noise
          # delta += diffusion * (1-t)**(0.5)

          if grad_mask is not None:
            delta = delta * grad_mask
            
          delta = drift * step_size
          x += delta

      else:

        def ode_func(t, x):

          # Prepare potential network input
          # t = t_z(z)
          # if sde.config.training.continuous:
          #   tao = t
          # else:
          #   tao = round(t,3)
          #   if tao == 0: tao = 5e-8
          #   elif tao == 1: tao = 1 - 5e-8

          if t > t_max:
            tao = t_max
          else:
            tao = t
          
          state['t_eval'] += 1; step_size = abs(t) / state['t_eval']

          samples_x = from_flattened_numpy(x, (sample_size, -1)).to(device).type(torch.float32)
          samples_x.requires_grad = True

          with torch.enable_grad():
            # Get model function
            net_fn = get_predict_fn(sde, model, train=False)

            # Predict scalar potential (FC)
            samples_net = samples_x
            if sde.config.training.augment_t:
              if sde.config.model.temb_type == 'time':
                cond = t
              elif sde.config.model.temb_type == 'lamb': 
                if sde.config.training.scheduling == 'cosine':
                  lamb = 2 * math.log(math.tan(tao*math.pi/2))
                elif sde.config.training.scheduling == 'fmot':
                  lamb = 1 * math.log((tao)/(1-tao))
                elif sde.config.training.scheduling == 'vapo':
                  lamb = 2 * math.log((tao)/(1-tao))
                elif sde.config.training.scheduling == 'elbo':
                  lamb = 1 * math.log((2*tao**2)/(1-tao)**2)
                if sde.config.training.shift_schedule:
                  lamb += math.log(sde.config.training.shift_ref / sde.config.data.image_size)
                cond = lamb
              cond_samples = cond * torch.ones(sample_size, 1).to(device).type(torch.float32)
              std_enc = (1-tao) * torch.ones(sample_size, 1).to(device).type(torch.float32)
              # temb = GaussianFourierProjection(sde.config.model.temb_dim, emb_type='fourier')(cond_samples)
              samples_net = torch.cat([samples_x, cond_samples, std_enc], dim=-1)
            if sde.config.training.class_guidance:
              # 1-airplane  2-automobile  3-bird  4-cat  5-deer  6-dog  7-frog  8-horse  9-ship  10-truck
              labels = torch.tensor([0,0,0,1,0,0,0,0,0,0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
              samples_net = torch.cat([samples_x, cond_samples, labels], dim=-1)
              labels_null = torch.tensor([0,0,0,0,0,0,0,0,0,0])[None,:].to(device).type(torch.float32).repeat(sample_size,1)
              samples_net_null = torch.cat([samples_x, cond_samples, labels_null], dim=-1)
              
            psi = net_fn(samples_net).squeeze(dim=-1)
            if sde.config.training.class_guidance:
              psi_null = net_fn(samples_net_null).squeeze(dim=-1)

            # Normalize field by its mean
            # psi -= psi.mean(dim=0, keepdim=True)

            # Compute (backpropagate) N-dimensional Poisson field (gradient)
            # print(sde.config.sampling.weight)
            drift = torch.autograd.grad(psi, samples_x, torch.ones_like(psi))[0]
            if sde.config.training.class_guidance:
              drift_null = torch.autograd.grad(psi_null, samples_x, torch.ones_like(psi))[0]
              w = sde.config.sampling.guidance_strength
              drift = (1 + w) * drift - w * drift_null

            # Normalize field
            # drift_norm =  torch.sqrt(torch.sum(drift**2, axis=-1, keepdims=True))
            # drift = drift / drift_norm
            # drift = drift * math.sqrt(data_dim)
            drift = drift * sde.config.sampling.weight

          if grad_mask is not None:
            drift = drift * grad_mask

          # print(z, psi.mean())
          return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE.
        # Note that we use z = exp(t) for change-of-variable to accelearte the ODE simulation
        # boundary = [np.log(sde.config.training.z_max), np.log(eps)]
        # boundary = [z_t(t_start), z_t(t_end)]
        boundary = [t_start, t_end]
        solution = integrate.solve_ivp(ode_func, boundary, to_flattened_numpy(x), rtol=rtol, atol=atol, method=method)

        nfe = solution.nfev
        x = torch.tensor(solution.y[:,-1]).reshape(sample_size, -1).to(device).type(torch.float32)

      # Detach augmented z dimension
      if inverse_scale: x = inverse_scaler(x.reshape(new_shape))
      if sde.config.data.image_size_ori < sde.config.data.image_size: x = resize_op(x)

      return x, nfe

  return ode_sampler