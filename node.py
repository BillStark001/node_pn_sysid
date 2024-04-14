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
      params1: torch.Tensor,
      params2: torch.Tensor,
    ):
    super(NeuralODEFunc, self).__init__()
    self.omega0 = omega
    self.L = L
    self.params1 = nn.Parameter(params1)
    self.params2 = nn.Parameter(params2)

  def forward(self, t, X):
    M1, D1, V1, Pmech1 = self.L
    M2_raw, D2_raw, V2 = self.params1
    Pmech2, G, B = self.params2
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
  solver_options: Optional[dict] = None,
  freq_factor: float = 0,
  freq_wnd: Optional[torch.Tensor] = None,
) -> Tensor:
  if solver_options is None:
    solver_options = {}
  X_pred = torchdiffeq.odeint_adjoint(node, X_init, t, **solver_options)
  
  X_pred_ = X_pred[..., :2]
  X_data_ = X_data[..., :2]

  loss_arr = (X_pred_ - X_data_) ** 2
  loss = torch.mean(loss_arr)
  
  freq_wnd = freq_wnd if freq_wnd is not None else 1
  if freq_factor > 0:
    F_data = torch.abs(torch.fft.rfft(X_data_ * freq_wnd, dim=0))
    F_pred = torch.abs(torch.fft.rfft(X_pred_ * freq_wnd, dim=0))
    loss_f_arr = (F_pred - F_data) ** 2
    loss_f = torch.mean(loss_f_arr)
    return loss + freq_factor * loss_f
  
  return loss