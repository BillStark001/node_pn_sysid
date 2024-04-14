from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchdiffeq

Tensor = torch.Tensor

class NeuralODEFunc(nn.Module):
  
  def __init__(
      self, 
      omega: torch.Tensor,
      L: torch.Tensor,
      params: torch.Tensor,
    ):
    super(NeuralODEFunc, self).__init__()
    self.omega0 = omega
    self.L = L
    self.params = nn.Parameter(params)

  def forward(self, t, X):
    M1, D1, V1, Pmech1 = self.L
    M2_raw, D2_raw, V2, Pmech2, G, B = self.params
    M2 = M2_raw # * 10
    D2 = D2_raw # * 10

    d1 = X[..., 0]
    o1 = X[..., 1]
    d2 = X[..., 2]
    o2 = X[..., 3]

    do1 = (
        (-D1) * o1
        + V1 * V2 * B * torch.sin(d1 - d2)
        - V1 * V2 * G * torch.cos(d1 - d2)
        + Pmech1
    ) / M1

    do2 = (
        (-D2) * o2
        + V1 * V2 * B * torch.sin(d2 - d1)
        - V1 * V2 * G * torch.cos(d2 - d1)
        + Pmech2
    ) / M2

    dX_dt = torch.stack([
      self.omega0 * o1, 
      do1, 
      self.omega0 * o2, 
      do2
    ]).T

    return dX_dt


def loss_func(
  node: NeuralODEFunc, t: Tensor, X_init: Tensor, X_data: Tensor,
  solver_options: Optional[dict] = None
) -> Tensor:
  if solver_options is None:
    solver_options = {}
  X_pred = torchdiffeq.odeint_adjoint(node, X_init, t, **solver_options)

  loss_arr = ((X_pred - X_data) ** 2)[..., :2]
  loss = torch.mean(loss_arr)
  return loss