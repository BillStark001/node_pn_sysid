from typing import Callable, Tuple
from numpy.typing import NDArray

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torchdiffeq

import node
import syntax_tree as st
from syntax_tree.miss_hit_helper import get_function_by_name, parse_matlab_code
from syntax_tree.src_exec_cfg import exec_func
from utils import DictWrapper, cache

@dataclasses.dataclass
class ScenarioParameters:
  params: dict
  inputs: dict
  network_path: str
  network_name: str
  
  t: NDArray
  x0: NDArray
  true_x: NDArray
  
  all_params: list
  normal_params: list
  special_params: list
  clamp_params: list
  
  observable_y_indices: NDArray
  
def default_optim_factory(func: torch.nn.Module, s_params: ScenarioParameters):
  normal_lr = 0.002
  special_lr = 0.0002
  normal_params_names = s_params.normal_params
  special_params_names = s_params.special_params
  
  normal_params = [param for name, param in func.named_parameters() \
    if name in normal_params_names]
  special_params = [param for name, param in func.named_parameters() \
    if name in special_params_names]
  param_groups = [
    {'params': normal_params, 'lr': normal_lr}, 
    {'params': special_params, 'lr': special_lr}
  ]
  optimizer = torch.optim.RMSprop(param_groups)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.5, verbose=False)
  return optimizer, scheduler

@dataclasses.dataclass
class SolverParameters:
  device: str
  optim_factory: Callable[[torch.nn.Module, ScenarioParameters], Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]
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


global_vars_dict = dict(
  sin = torch.sin,
  cos = torch.cos,
)

class ODEFunc(nn.Module):

  def __init__(
    self, 
    func: Callable[[dict, dict], torch.Tensor],
    params: dict,
    inputs_key = 'y',
  ):
    super().__init__()
    self.func = func
    self.params = params
    self.inputs_key = inputs_key
    
  def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    for name, param in self.params.items():
      if isinstance(param, nn.Parameter):
        yield name, param

  def parameters(self, recurse: bool = True):
    for name, param in self.params.items():
      if isinstance(param, nn.Parameter):
        yield param
    
  def forward(self, t: float, y: torch.Tensor):
    with torch.enable_grad():
      inputs = { self.inputs_key: y }
      res: torch.Tensor = self.func(inputs, self.params)
      return res.reshape(-1)
  
class EnvModelEstimator(object):
  
  def __init__(
    self, 
    params: ScenarioParameters,
    sol_params: SolverParameters,
  ):
    self.params = params
    self.sol_params = sol_params
    
    self.func: torch.nn.Module = None
    self.optimizer: torch.optim.Optimizer = None
    self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
    
    self.t: torch.Tensor = None
    self.true_x0: torch.Tensor = None
    self.true_x: torch.Tensor = None
    self.data_size = 0
    
    self.observable_indices = np.arange(0, 2, dtype=int)
    self.clamp_params: list = []
    
    
  def init(self):
    
    self.device = self.sol_params.device or 'cpu'
    
    # params
    
    p = self.params
    with open(p.network_path, 'r', encoding="utf-8") as f:
      nw_func_content = f.read()
    file_ast = parse_matlab_code(nw_func_content, p.network_path)
    func_ast = get_function_by_name(file_ast)
      
    # create residual function
    def residual_function(inputs: dict, params: dict):
      dydt = exec_func(
        func_ast,
        [
          DictWrapper(inputs),
          DictWrapper(params),
        ],
        global_vars_dict
      )
      return dydt
    
    # transform params from numpy arrays to trainable parameters
    all_params = {**p.params}      
    train_params = set(self.params.normal_params + self.params.special_params)
    for k, v in all_params.items():
      vn = torch.from_numpy(v)
      if k in train_params:
        all_params[k] = nn.Parameter(vn)
      else:
        all_params[k] = vn
        
    all_inputs = { k: torch.from_numpy(v) for k, v in p.inputs.items() }
    
    # trace it a priori
    traced_residual_function = torch.jit.trace(
      residual_function, (all_inputs, all_params),
    )
    # TODO load trace from file
    
    # compose function
    self.func = ODEFunc(
      traced_residual_function if traced_residual_function is not None else residual_function,
      all_params,  
    ).to(self.device)
    
    self.optimizer, self.scheduler = self.sol_params.optim_factory(self.func, p)
    
    self.t = torch.from_numpy(self.params.t).to(self.device)
    self.true_x = torch.from_numpy(self.params.true_x).to(self.device)
    self.true_x0 = torch.from_numpy(self.params.x0)
    
    self.observable_indices = p.observable_y_indices
    self.clamp_params = p.clamp_params
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
      loss = torch.mean((
        pred_x[..., self.observable_indices] - \
          batch_x[..., batch_n, self.observable_indices]) ** 2) 
      loss.backward() #Calculate the dloss/dparameters
      self.optimizer.step() #Update value of parameters

      with torch.no_grad():
        for name, param in self.func.named_parameters():
          if name in self.clamp_params:
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
      loss = torch.mean((pred_x[:,0:2] - self.true_x[:,0:2])**2) 
    return loss.item()
  
@cache('./run/solver.pkl')
def create_estimator(*args, **kwargs):
  return EnvModelEstimator(*args, **kwargs)
  
  