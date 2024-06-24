import numpy as np

import torch
import torch.nn as nn


class MatlabTracedModule(nn.Module):
  
  def __init__(self, uuid_list, graph):
    super().__init__()
    self.uuid_list = uuid_list
    self.graph = graph
    
  def forward(self, *args):
    replace_dict = {
      self.uuid_list[i]: args[i] for i in range(len(self.uuid_list))
    }
    return eval_m(self.graph, replace_dict)
  

class MatlabWrappedModule(nn.Module):

  def __init__(
    self, 
    graph, params, param_uuids, input_uuids, 
    traced_model=None
  ):
    super().__init__()
    self.graph: dict = graph
    self.params: dict = params
    self.param_uuids: dict = param_uuids
    
    self.inputs = []
    self.uuid_tensor_map = {}
    for k, uuid in input_uuids.items():
      self.inputs.append(k)
      self.uuid_tensor_map[uuid] = None
    for k, uuid in param_uuids.items():
      self.uuid_tensor_map[uuid] = params[k]
    
    # input_uuids and then param uuids
    self.uuid_order = list(self.uuid_tensor_map.keys())
    
    self.traced_model = traced_model
    if traced_model is None:
      self.traced_model = MatlabTracedModule(self.uuid_order, graph)

  def get_traced_module_input(self, *inputs):
    w_list = [self.uuid_tensor_map.get(uuid, None) \
      for uuid in self.uuid_order]
    for i in range(len(inputs)):
      w_list[i] = inputs[i]
    return w_list

  def forward(self, *inputs):
    inputs_ = self.get_traced_module_input(*inputs)
    return self.traced_model(*inputs_)

  def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    for name, param in self.params.items():
      yield name, param

  def parameters(self, recurse: bool = True):
    for name, param in self.params.items():
      yield param


def eval_m(n, replace) -> torch.Tensor:

  if not isinstance(n, dict):
    if isinstance(n, (str, int, float, np.ndarray)):
      return n
    if isinstance(n, list):
      return [eval_m(i, replace) for i in n]
    raise Exception('What the hell?')

  uuid = n['Uuid']
  if uuid in replace:
    return replace[uuid]

  type = n['Type']
  data = n['Data']  # array or str
  nc = n['Nodes']

  def _child(k): return eval_m(nc[k], replace)
  ret = None

  if type == 'opr':

    if data == 'plus':
      ret = _child(0) + _child(1)
    elif data == 'minus':
      ret = _child(0) - _child(1)

    elif data == 'uplus':
      ret = _child(0)
    elif data == 'uminus':
      ret = -_child(0)

    elif data == 'times':
      ret = _child(0) * _child(1)
    elif data == 'mtimes':
      ret = _child(0) @ _child(1)
    elif data == 'rdivide':
      ret = _child(0) / _child(1)
    elif data == 'ldivide':
      ret = _child(1) / _child(0)
    elif data == 'mrdivide':
      ret = _child(0) @ torch.inverse(_child(1))
    elif data == 'mldivide':
      ret = torch.inverse(_child(1)) @ _child(0)

    elif data == 'power':
      ret = _child(0) ** _child(1)
    elif data == 'mpower':
      ret = torch.linalg.matrix_power(_child(0), _child(1))

    elif data == 'lt':
      ret = _child(0) < _child(1)
    elif data == 'gt':
      ret = _child(0) > _child(1)
    elif data == 'le':
      ret = _child(0) <= _child(1)
    elif data == 'ge':
      ret = _child(0) >= _child(1)
    elif data == 'ne':
      ret = _child(0) != _child(1)
    elif data == 'eq':
      ret = _child(1) == _child(0)

    elif data == 'and':
      ret = torch.logical_and(_child(0), _child(1))
    elif data == 'or':
      ret = torch.logical_or(_child(0), _child(1))
    elif data == 'not':
      ret = torch.logical_not(_child(0))

    elif data in {'ctranspose', 'transpose'}:
      ret = torch.transpose(_child(0), 0, 1)

    elif data == 'horzcat':
      ret = torch.hstack([eval_m(x, replace) for x in nc])
    elif data == 'vertcat':
      ret = torch.vstack([eval_m(x, replace) for x in nc])

    elif data in {'colon2', 'colon3'}:
      raise Exception('Not Implemented: ' + data)
    
    elif data == 'end_index':
      node = _child(0) # must be torch tensor
      ind_dim = int(_child(1)[0, 0]) - 1
      # n_dims = int(_child(2)[0, 0])
      len_dim = node.shape[ind_dim]
      ret = len_dim

    elif data in {'subsref_arr', 'subsasgn_arr'}:
      node = _child(0)
      subs = _child(1)
      is_assign = data == 'subsasgn_arr'
      asgn_target = None if not is_assign else _child(2)
      subs_parsed = []
      for sub in subs:
        sub_parsed = None
        if sub == ':':
          sub_parsed = slice(None)
        elif isinstance(sub, np.ndarray):
          sub_int = sub.astype(int) - 1 # to align python indices
          if sub_int.size == 1:
            sub_int = sub_int[0][0]
            sub_parsed = slice(sub_int, sub_int + 1)
          else: # sub_int is an index slice
            sub_parsed = sub_int
        else:
          raise Exception('Not Implemented: subsref_arr - ' + sub)
        subs_parsed.append(sub_parsed)
        
      if len(subs_parsed) == 1:
        if node.ndim > 2:
          dim_t = tuple(node.shape[:node.ndim - 2])
          nv = node.view((*dim_t, -1))
        else:
          nv = node.view(-1)
      else:
        nv = node
        
      if is_assign:
        nv[..., *subs_parsed] = asgn_target
        ret = node
      else:
        ret = nv[..., *subs_parsed]
        if ret.ndim == 1:
          ret = ret.unsqueeze(1)
      

    # TODO subsref, subsasgn, colon
    else:
      raise Exception('Not Implemented: ' + data)

  elif type == 'var':
    ret = torch.from_numpy(data)

  elif type == 'func':
    ret = getattr(torch, data)(*[eval_m(x, replace) for x in nc])

  if ret == None:
    raise Exception('What the hell, again???')
  replace[uuid] = ret
  return ret