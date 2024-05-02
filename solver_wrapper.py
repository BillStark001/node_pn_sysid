from typing import Callable, Tuple
from numpy.typing import NDArray

import dataclasses
import numpy as np
import torch
import torchdiffeq

import node

@dataclasses.dataclass
class ScenarioParameters:
  omega: float
  
  M_1: float
  D_1: float
  V_field_1: float
  P_mech_1: float
  
  M_2: float
  D_2: float
  V_field_2: float
  P_mech_2: float
  
  G_12: float
  B_12: float
  
  t: NDArray
  true_x: NDArray
  
def default_optim_factory(func: torch.nn.Module):
  normal_lr = 0.001
  special_lr = 0.0001
  special_param = [func.Pmech2, func.B, func.G]
  other_param = [param for name, param in func.named_parameters() if param not in special_param]
  param_groups = [{'params': other_param, 'lr': normal_lr}, {'params': special_param, 'lr': special_lr}]
  optimizer = torch.optim.RMSprop(param_groups) #RMSprop
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.5, verbose=False)
  return optimizer, scheduler

@dataclasses.dataclass
class SolverParameters:
  device: str
  optim_factory: Callable[[torch.nn.Module], Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]
  batch_time: int
  batch_size: int
  ode_params: dict
  
  
  

def get_batch(
  t: NDArray,
  data_size: int,
  true_x: NDArray,
  batch_time: int,
  batch_size: int,
  device = 'cpu'
):
  s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
  batch_x0 = true_x[s]
  batch_t = t[:batch_time]
  batch_x = torch.stack([true_x[s + i] for i in range(batch_time)], dim=0)
  return s.to(device), batch_x0.to(device), batch_t.to(device), batch_x.to(device)
  
class EnvModelEstimator(object):
  
  def __init__(
    self, 
    params: ScenarioParameters,
    sol_params: SolverParameters,
  ):
    self.params = params
    self.sol_params = sol_params
    
    self.func: node.NODEMechanizedMoris = None
    self.optimizer: torch.optim.Optimizer = None
    self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
    
    self.t: torch.Tensor = None
    self.true_x0: torch.Tensor = None
    self.true_x: torch.Tensor = None
    self.data_size = 0
    
    
  def init(self):
    p = self.params
    self.device = self.sol_params.device or 'cpu'
    _t = torch.tensor
    self.func = node.NODEMechanizedMoris(
      _t(p.omega),
      _t(p.M_1),
      _t(p.D_1),
      _t(p.M_2),
      _t(p.D_2),
      _t(p.V_field_1),
      _t(p.V_field_2),
      _t(p.B_12),
      _t(p.G_12),
      _t(p.P_mech_1),
      _t(p.P_mech_2),
    ).to(self.device)
    self.optimizer, self.scheduler = self.sol_params.optim_factory(self.func)
    self.t = torch.from_numpy(self.params.t).to(self.device)
    self.true_x = torch.from_numpy(self.params.true_x).to(self.device)
    self.true_x0 = self.true_x[0]
    self.data_size = self.params.t.size
    
    
  def iterate(self):
    self.optimizer.zero_grad()
    s, batch_x0, batch_t, batch_x = get_batch(
      self.t,
      self.data_size,
      self.true_x,
      self.sol_params.batch_time,
      self.sol_params.batch_size,
      self.device
    )
    
    grad_norm_acc = 0
    loss_acc = 0
    
    batch_size = self.sol_params.batch_size

    for batch_n in range(0, batch_size):
      
      pred_x = torchdiffeq.odeint_adjoint(
        self.func, batch_x0[batch_n], batch_t, 
        **(self.sol_params.ode_params or {})).to(self.device)
      loss = torch.mean((pred_x[..., 0:2] - batch_x[..., batch_n, 0:2])**2) 
      loss.backward() #Calculate the dloss/dparameters
      self.optimizer.step() #Update value of parameters

      with torch.no_grad():
        for name, param in self.func.named_parameters():
          if name in ['M2', 'D2', 'V2', 'Pmech2']:
            param.clamp_(min=0.1) #make sure values are positive

      for param in self.func.parameters():
        if param.grad is not None:
          grad_norm = param.grad.data.norm(2).item() ** 2
      grad_norm_acc += np.sqrt(grad_norm)
      
      loss_acc += loss.item()
      
    self.scheduler.step()
    
    return loss_acc / batch_size, grad_norm / batch_size
    
    
  def get_current_params(self):
    ret = {}
    with torch.no_grad():
      for name, param in self.func.named_parameters():
        ret[name] = param.cpu().numpy()
    return ret


  def evaluate(self):
    with torch.no_grad():
      pred_x = torchdiffeq.odeint_adjoint(
        self.func, self.true_x0, self.t).to(self.device)
      loss = torch.mean((pred_x[:,0:2,0] - self.true_x[:,0:2,0])**2) 
    return loss.item()
  
  