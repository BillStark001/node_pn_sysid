from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchdiffeq

Tensor = torch.Tensor

class NODEMechanized(nn.Module):
  
  def __init__(
      self, 
      omega: torch.Tensor,
      L: torch.Tensor,
      params0: torch.Tensor,
      params1: torch.Tensor,
      params2: torch.Tensor,
    ):
    super().__init__()
    self.omega0 = omega
    self.L = L
    self.params0 = nn.Parameter(params0)
    self.params1 = nn.Parameter(params1)
    self.params2 = nn.Parameter(params2)

  def forward(self, t, X):
    M1, D1, V1, Pmech1 = self.L
    
    M2, D2 = self.params0
    V2, Pmech2 = self.params1
    G, B = self.params2
  
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


class NODEMechanizedMoris(nn.Module):
  def __init__(
    self, 
    omega: torch.Tensor,
    M1, D1, M2, D2, V1, V2, B, G, Pmech1, Pmech2,
    ):
    super().__init__()
    self.omega0 = omega
    self.M1 = M1
    self.M2 = nn.Parameter(M2)
    self.D1 = D1
    self.D2 = nn.Parameter(D2)
    self.V1 = V1
    #self.V2 = V2
    self.V2 = nn.Parameter(V2)
    #self.B = B
    self.B = nn.Parameter(B)
    #self.G = G
    self.G = nn.Parameter(G)
    self.Pmech1 = Pmech1
    #self.Pmech2 = Pmech2
    self.Pmech2 = nn.Parameter(Pmech2)
      
  def forward(self, t, y):
    dydt = torch.zeros_like(y)
    omega0 = self.omega0
    dydt[0] = y[1]
    dydt[1] = (-self.D1*y[1]/omega0 + self.V1*self.V2*self.B*torch.sin(y[0]-y[2]) - self.V1*self.V2*self.G*torch.cos(y[0]-y[2]) + self.Pmech1)*omega0/self.M1
    dydt[2] = y[3]
    dydt[3] = (-self.D2*y[3]/omega0 + self.V1*self.V2*self.B*torch.sin(y[2]-y[0]) - self.V1*self.V2*self.G*torch.cos(y[2]-y[0]) + self.Pmech2)*omega0/self.M2
    self.dydt = dydt
    return dydt

class NODENeural(nn.Module):

  def __init__(self, omega: torch.Tensor):
    super().__init__()
    self.omega0 = omega

    self.net = nn.Sequential(
        nn.Linear(4, 25),
        nn.Tanh(),
        nn.Linear(25, 25),
        nn.Tanh(),
        nn.Linear(25, 25),
        nn.Tanh(),
        nn.Linear(25, 2),
    )

    for m in self.net.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.normal_(m.bias, mean=0, std=0.05)

  def forward(self, t, y):
    delta = y[..., 0::2]
    omega = y[..., 1::2]
    dydt = torch.zeros_like(y)
    dydt[..., 0::2] = omega * self.omega0
    # y_feed = torch.concat([
    #   y,
    #   torch.sin(delta[..., 0:1] - delta[..., 1:2]),
    #   torch.cos(delta[..., 0:1] - delta[..., 1:2])
    # ], dim=-1)
    # dydt_pred = self.net(y_feed)
    dydt_pred = self.net(y)
    dydt[..., 1::2] = dydt_pred
    return dydt


def loss_func(
  node: nn.Module, t: Tensor, X_init: Tensor, X_data: Tensor,
  solver_options: Optional[dict] = None,
  time_weight: Optional[torch.Tensor] = None,
  freq_weight: float = 0,
  freq_wnd: Optional[torch.Tensor] = None,
) -> Tensor:
  if solver_options is None:
    solver_options = {}
  X_pred = torchdiffeq.odeint_adjoint(node, X_init, t, **solver_options)
  
  time_weight = time_weight if time_weight is not None else 1
  X_pred_ = X_pred[..., :2]
  X_data_ = X_data[..., :2]

  loss_arr = (X_pred_ - X_data_) ** 2
  loss = torch.mean(loss_arr * time_weight)
  
  freq_wnd = freq_wnd if freq_wnd is not None else 1
  if freq_weight > 0:
    F_data = torch.abs(torch.fft.rfft(X_data_ * freq_wnd, dim=0))
    F_pred = torch.abs(torch.fft.rfft(X_pred_ * freq_wnd, dim=0))
    loss_f_arr = (F_pred - F_data) ** 2
    loss_f = torch.mean(loss_f_arr)
    return (loss + freq_weight * loss_f), X_pred
  
  return loss, X_pred