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
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, loss_type="original"):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

      if loss_type == "new":
        with torch.no_grad():
            batch_size = batch.shape[0] // 2
            dim = perturbed_data.shape[1] * perturbed_data.shape[2] * perturbed_data.shape[3]
            xt = perturbed_data[:batch_size].view(batch_size, -1)  # (batch_size, -1)
            x0 = batch[:batch_size].view(xt.shape[0], -1)  # (batch_size, -1)
            alpha_t = torch.ones(batch_size, 1, device=batch.device)  # (batch_size, 1)
            sigma_t = std[:batch_size, None]  # (batch_size, 1)

            weight = -0.5 * (xt.unsqueeze(1) - alpha_t.unsqueeze(1)*x0.unsqueeze(0)).pow(2).sum(-1,True) / sigma_t.unsqueeze(1)**2
            weight_normalized = torch.softmax(weight, 1)  # (batch_size, batch_size, 1)

            score_cond = - (xt.unsqueeze(1) - alpha_t.unsqueeze(1)*x0.unsqueeze(0)) / sigma_t.unsqueeze(1)**2  # (batch_size, batch_size, 2)
            score_gt = (weight_normalized * score_cond).sum(1)  # (batch_size, 2)

            score_score_gt = (weight_normalized * score_cond.pow(2).sum(-1,True)).sum(1)  # (batch_size, 1)

            x0_2 = batch[batch_size:-1].view(batch_size-1, -1)  # (batch_size, -1)
            x0_2 = torch.cat([x0.unsqueeze(1), x0_2.unsqueeze(0).repeat(batch_size,1,1)], 1)  # (batch_size, batch_size, 2)

            weight = -0.5 * (xt.unsqueeze(1) - alpha_t.unsqueeze(1)*x0_2).pow(2).sum(-1,True) / sigma_t.unsqueeze(1)**2
            weight_normalized = torch.softmax(weight, 1)  # (batch_size, batch_size, 1)
            score_cond = - (xt.unsqueeze(1) - alpha_t.unsqueeze(1)*x0_2) / sigma_t.unsqueeze(1)**2  # (batch_size, batch_size, 2)
            score_gt_2 = (weight_normalized * score_cond).sum(1)  # (batch_size, 2)

            cov_trace = score_score_gt - (score_gt*score_gt_2).sum(-1,True)  # (batch_size, 1)
            hessian_trace = - torch.ones_like(cov_trace)  # (batch_size, 1)
            score_gt_jvp_trace = hessian_trace + cov_trace * sigma_t**2 / dim

        # v = torch.sign(torch.randn_like(perturbed_data[:batch_size]))
        # h = 0.01
        # score_jvp = (score_fn(perturbed_data[:batch_size] + h * v, t[:batch_size]) - score[:batch_size].detach()) / h
        # score_jvp_trace = (score_jvp * v).sum((1,2,3))[:, None]
        perturbed_data.requires_grad = True
        score = score_fn(perturbed_data, t)
        v = torch.sign(torch.randn_like(score))
        score_vjp = torch.autograd.grad(
          outputs=score, inputs=perturbed_data, grad_outputs=v, 
          create_graph=True,
        )[0]
        dim = perturbed_data.shape[1] * perturbed_data.shape[2] * perturbed_data.shape[3]
        score_jvp_trace = (score_vjp * v).sum((1,2,3))[:batch_size, None]

        losses2 = torch.square(score_jvp_trace * sigma_t**2 / dim - score_gt_jvp_trace)
        losses2 = 0.5 * reduce_op(losses2, dim=-1)
      else:
        perturbed_data.requires_grad = True
        score = score_fn(perturbed_data, t)

        v = torch.sign(torch.randn_like(score))
        score_vjp = torch.autograd.grad(
          outputs=score, inputs=perturbed_data, grad_outputs=v, 
          create_graph=True,
        )[0]
        dim = perturbed_data.shape[1] * perturbed_data.shape[2] * perturbed_data.shape[3]
        div_score = (score_vjp * v).sum((1,2,3))

      if loss_type == "mse":
        losses2 = losses*0

      elif loss_type == "original":
        l1 = (score * std[:,None,None,None] + z).detach()
        losses2_frob = torch.square(
          score_vjp * std[:,None,None,None]**2 + v - (l1 * v).sum((1,2,3),True) * l1
        ) / dim
        losses2_div = torch.square(
          div_score * std**2 + dim - (l1*v).sum((1,2,3)).pow(2)
        ) / dim**2
        losses2_frob = reduce_op(losses2_frob.reshape(losses2_frob.shape[0], -1), dim=-1)
        losses2_div = reduce_op(losses2_div.reshape(losses2_div.shape[0], -1), dim=-1)
        losses2 = losses2_frob + losses2_div

      elif loss_type == "proposed":
        losses2 = 0.01 * torch.square(div_score * std**2 + dim * torch.ones_like(div_score)) / dim

    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses), torch.mean(losses2)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, loss_type="original"):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting, loss_type=loss_type)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      sum(loss).backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
