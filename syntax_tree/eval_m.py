import numpy as np

import torch
import torch.nn as nn

from syntax_tree.eval_opr import eval_subs_ref_arr

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


    # arithmetic
    
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

    elif data in {'ctranspose', 'transpose'}:
      ret = torch.transpose(_child(0), 0, 1)

    # compare


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
      
    # logical

    elif data == 'and':
      ret = torch.logical_and(_child(0), _child(1))
    elif data == 'or':
      ret = torch.logical_or(_child(0), _child(1))
    elif data == 'not':
      ret = torch.logical_not(_child(0))

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
      subs = [slice(None) if sub == ':' else sub for sub in _child(1)]
      is_assign = data == 'subsasgn_arr'
      asgn_target = None if not is_assign else _child(2)
      ret = eval_subs_ref_arr(node, subs, asgn_target)
      

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