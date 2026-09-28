"""First-order (no double backprop) training for the VAPO loss.

The VAPO loss depends on grad_x Phi (and dPhi/dt) through the gradient-norm, cosine and time-derivative
terms, so the reference implementation calls torch.autograd.grad(..., create_graph=True) and then
back-propagates through that backward graph (reverse-over-reverse).  This module computes the SAME
parameter gradient using only ordinary first-order backward passes:

  * For any loss l(g) of g_i = grad_(x,t) Phi(x_i, t_i),
        grad_theta l = sum_i U_i . grad_theta g_i,          U_i = dl/dg_i  (a detached vector)
                     = grad_theta sum_i ||U_i|| * D_{U_i/||U_i||} Phi(x_i, t_i),
    i.e. one directional derivative of Phi per sample along a *fixed* direction.  It is evaluated by a
    central finite difference [Phi(z + h u) - Phi(z - h u)] / (2h) (error O(h^2)), which only needs
    forward passes of Phi and first-order backward passes.
  * With the detached normaliser of the covariance/correlation term, its gradient is
    grad_theta sum_i a_i Phi(x_i) with detached weights a_i.  Because Phi_i depends only on x_i
    (GroupNorm, attention and dropout act per sample), ONE backward pass with cotangent a yields both
    that parameter gradient and a_i * grad_x Phi(x_i), from which g_i is recovered.
  * The whole surrogate is a weighted sum of per-sample Phi evaluations, so the three forward passes
    are back-propagated one at a time: peak activation memory equals ordinary first-order training.

Requirements / caveats:
  * Dropout masks must be identical in all three passes (handled here by replaying the RNG state).
  * The network must be per-sample independent (no BatchNorm in train mode).
  * The finite difference needs full fp32: disable TF32 (torch.backends.cudnn.allow_tf32 = False,
    torch.backends.cuda.matmul.allow_tf32 = False) and do not autocast these passes to fp16/bf16.
    Run check_first_order.py to confirm agreement with double backprop on your hardware/precision.
"""

import torch


def _get_rng_state():
  cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  return torch.get_rng_state(), cuda


def _set_rng_state(state):
  cpu, cuda = state
  torch.set_rng_state(cpu)
  if cuda is not None:
    torch.cuda.set_rng_state_all(cuda)


def first_order_backward(net_fn, x, cond, extra, zeroth_weights, grad_loss_fn, fd_step=3e-2):
  """Accumulates grad_theta of  sum_i zeroth_weights_i * Phi_i + grad_loss_fn(g_x, g_t)  into `.grad`.

  Args:
    net_fn: callable mapping the network input (concatenated [x, cond, extra] as in losses.py) to Phi, shape [B].
    x: [B, D] perturbed samples (detached).
    cond: [B, 1] time input that is differentiated (cond_samples in losses.py), or None if not augmented.
    extra: list of [B, k] tensors appended after cond that are NOT differentiated (e.g. std_enc, labels).
    zeroth_weights: callable psi_detached -> [B] detached weights a_i such that the zeroth-order part of the
      loss has gradient grad_theta sum_i a_i Phi_i (e.g. the covariance/correlation term).
    grad_loss_fn: callable (g_x, g_t) -> scalar loss built from the gradients (g_t is None if cond is None).
      It is evaluated on detached leaves, so it must be differentiable w.r.t. its inputs only.
    fd_step: finite-difference step along the unit direction in (x, t) space.

  Returns:
    psi (detached [B]), g_x (detached), g_t (detached or None), grad_loss (detached scalar).
  """
  def inputs(xx, cc):
    parts = [xx] + ([cc] if cc is not None else []) + list(extra)
    return torch.cat(parts, dim=-1) if len(parts) > 1 else xx

  x = x.detach().requires_grad_(True)
  cond = cond.detach().requires_grad_(True) if cond is not None else None
  rng_pre = _get_rng_state()

  # Pass 1: Phi(x); a single backward with cotangent a gives the zeroth-order parameter gradient AND a_i * g_i.
  psi = net_fn(inputs(x, cond)).reshape(-1)
  rng_post = _get_rng_state()
  a = zeroth_weights(psi.detach()).detach()
  a_max = a.abs().max()
  tiny = torch.finfo(a.dtype).tiny
  if a_max > 0:
    # Floor |a_i| at 1e-6 max|a| so that a_i * g_i stays well above underflow (changes the gradient by <= 1e-6).
    floor = a_max * 1e-6
    a = torch.where(a.abs() >= floor, a, torch.where(a >= 0, floor, -floor))
    torch.autograd.backward(psi, grad_tensors=a)
    g_x = (x.grad / a[:, None]).detach()
    g_t = (cond.grad / a[:, None]).detach() if cond is not None else None
  else:  # no zeroth-order term: plain input gradients
    grads = torch.autograd.grad(psi, [x] + ([cond] if cond is not None else []), torch.ones_like(psi))
    g_x, g_t = grads[0].detach(), (grads[1].detach() if cond is not None else None)
  psi = psi.detach()

  # Direction U = d(grad loss)/dg on detached leaves (elementwise graph only, no network).
  g_x_leaf = g_x.clone().requires_grad_(True)
  g_t_leaf = g_t.clone().requires_grad_(True) if g_t is not None else None
  with torch.enable_grad():
    grad_loss = grad_loss_fn(g_x_leaf, g_t_leaf)
  leaves = [g_x_leaf] + ([g_t_leaf] if g_t_leaf is not None else [])
  us = torch.autograd.grad(grad_loss, leaves, allow_unused=True)
  us = [torch.zeros_like(l) if u is None else u for u, l in zip(us, leaves)]
  u_norm = torch.cat(us, dim=-1).norm(dim=-1).clamp_min(tiny)
  u_x = us[0] / u_norm[:, None]
  u_t = us[1] / u_norm[:, None] if cond is not None else None
  weight = u_norm / (2 * fd_step)

  # Passes 2-3: central difference, each back-propagated (and freed) on its own, same dropout masks.
  x0 = x.detach()
  c0 = cond.detach() if cond is not None else None
  for sign in (1.0, -1.0):
    _set_rng_state(rng_pre)
    xs = x0 + sign * fd_step * u_x
    cs = c0 + sign * fd_step * u_t if cond is not None else None
    phi = net_fn(inputs(xs, cs)).reshape(-1)
    torch.autograd.backward(phi, grad_tensors=sign * weight)
    del phi
  _set_rng_state(rng_post)
  return psi, g_x, g_t, grad_loss.detach()

