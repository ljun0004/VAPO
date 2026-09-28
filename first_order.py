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
    grad_theta sum_i a_i Phi(x_i) with detached weights a_i; it is carried by the same two passes via
    (Phi(z + h u) + Phi(z - h u)) / 2 = Phi(z) + O(h^2).  The first pass only computes grad_(x,t) Phi.
  * The whole surrogate is a weighted sum of per-sample Phi evaluations, so the three forward passes
    are back-propagated one at a time and no backward graph is ever kept: peak memory is about half that
    of the create_graph path (and ~1.35x a plain first-order step), for ~1.33x its FLOPs.

Requirements / caveats:
  * Dropout masks must be identical in all three passes (handled here by replaying the RNG state).
  * The network must be per-sample independent (no BatchNorm in train mode), since each sample gets
    its own finite-difference direction.
  * The finite difference needs full fp32: TF32 is disabled inside these passes unless allow_tf32=True,
    and they must not be autocast to fp16/bf16.
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


class _StrictFP32:
  """Disables TF32 inside the block (the finite difference cancels ~h of Phi's leading digits)."""

  def __init__(self, enabled):
    self.enabled = enabled

  def __enter__(self):
    self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    if self.enabled:
      torch.backends.cuda.matmul.allow_tf32 = False
      torch.backends.cudnn.allow_tf32 = False

  def __exit__(self, *exc):
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.saved


def first_order_backward(net_fn, x, cond, extra, zeroth_weights, grad_loss_fn, fd_step=3e-2,
                         allow_tf32=False):
  with _StrictFP32(not allow_tf32):
    return _first_order_backward(net_fn, x, cond, extra, zeroth_weights, grad_loss_fn, fd_step)


def _first_order_backward(net_fn, x, cond, extra, zeroth_weights, grad_loss_fn, fd_step):
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
    allow_tf32: keep TF32 convolutions/matmuls on (faster on Ampere+, but the finite difference then
      needs a larger fd_step and has ~% level error; check with check_first_order.py --allow_tf32).

  Returns:
    psi (detached [B]), g_x (detached), g_t (detached or None), grad_loss (detached scalar).
  """
  def inputs(xx, cc):
    parts = [xx] + ([cc] if cc is not None else []) + list(extra)
    return torch.cat(parts, dim=-1) if len(parts) > 1 else xx

  x = x.detach().requires_grad_(True)
  cond = cond.detach().requires_grad_(True) if cond is not None else None
  rng_pre = _get_rng_state()

  # Pass 1: Phi(x) and its input gradients only (no parameter gradients); the graph is freed here.
  psi = net_fn(inputs(x, cond)).reshape(-1)
  rng_post = _get_rng_state()
  grads = torch.autograd.grad(psi, [x] + ([cond] if cond is not None else []), torch.ones_like(psi))
  g_x = grads[0].detach()
  g_t = grads[1].detach() if cond is not None else None
  psi = psi.detach()
  a = zeroth_weights(psi).detach()
  tiny = torch.finfo(psi.dtype).tiny

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
  # The zeroth-order term rides along as (Phi(z+) + Phi(z-)) / 2 = Phi(z) + O(h^2).
  x0 = x.detach()
  c0 = cond.detach() if cond is not None else None
  for sign in (1.0, -1.0):
    _set_rng_state(rng_pre)
    xs = x0 + sign * fd_step * u_x
    cs = c0 + sign * fd_step * u_t if cond is not None else None
    phi = net_fn(inputs(xs, cs)).reshape(-1)
    torch.autograd.backward(phi, grad_tensors=0.5 * a + sign * weight)
    del phi
  _set_rng_state(rng_post)
  return psi, g_x, g_t, grad_loss.detach()

