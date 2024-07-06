from typing import Union, Literal, Callable, List, Dict

import torch.nn as nn

from miss_hit_core.m_ast import *
from syntax_tree.src_exec import exec_func


class MatlabSourceTraceModule(nn.Module):

  def __init__(self, src: Function_Definition, scope=None):
    super().__init__()
    self.src = src
    self.scope = scope

  def forward(self, *args):
    return exec_func(self.src, args, self.scope)


class MatlabSourceModule(nn.Module):

  def __init__(
      self,
      src: Function_Definition,
      params: dict,
      assign_params: Union[
          None,
          Literal['as_args', 'as_dict'],
          Callable[[List, Dict], List]
      ] = None,
      traced_model=None
  ):
    super().__init__()
    self.src = src
    self.params: dict = params

    # parse parameter assignment principle
    self.assign_params = assign_params if assign_params is not None else 'as_args'
    self.params_record = {}
    if self.assign_params == 'as_args':
      for i, n in enumerate(src.n_sig.l_inputs):
        p_name: str = n.t_ident.value
        if p_name in self.params:
          self.params_record[p_name] = i
    elif self.assign_params == 'as_dict':
      i = [i for i, n in enumerate(
          src.n_sig.l_inputs) if n.t_ident.value == 'params']
      assert len(i) > 0, '`as_dict` requires argument `params`'
      self.params_record['params'] = i[0]
    elif not callable(self.assign_params):
      raise ValueError(
          'Unrecognized parameter assignment principle: ' + self.assign_params)

    self.traced_model = traced_model
    if traced_model is None:
      self.traced_model = MatlabSourceTraceModule(src)

  def get_traced_module_input(self, *inputs):
    inputs_list = list(inputs)
    if callable(self.assign_params):
      return self.assign_params(inputs_list, self.params)
    # else use params_record
    for n, i in self.params_record.items():
      inputs_list.insert(i, self.params[n])
    return inputs_list

  def forward(self, *inputs):
    inputs_ = self.get_traced_module_input(*inputs)
    return self.traced_model(*inputs_)

  def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    for name, param in self.params.items():
      yield name, param

  def parameters(self, recurse: bool = True):
    for name, param in self.params.items():
      yield param
