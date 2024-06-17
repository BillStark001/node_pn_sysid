import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleFCN(nn.Module):
  def __init__(self):
    super(SimpleFCN, self).__init__()
    self.fc1 = nn.Linear(10, 50)
    self.fc2 = nn.Linear(50, 20)
    self.fc3 = nn.Linear(20, 1)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x


class MatlabWrappedModule(nn.Module):

  def __init__(self, params, forward):
    super().__init__()
    self.params = params
    self.forward_ = forward

  def forward(self, x):
    return self.forward_(dict(x=x), self.params)

  def named_parameters(self, prefix: str = '', recurse: bool = True):
    for name, param in self.params.items():
      yield name, param

  def parameters(self, recurse: bool = True):
    for name, param in self.params.items():
      yield param


def eval_m(n, replace) -> torch.Tensor:

  if not isinstance(n, dict):
    if isinstance(n, (str, int, float, np.ndarray)):
      return n
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

    elif data in {'subsindex', 'colon'}:
      raise Exception('Unsupported')

    # TODO subsref, subsasgn
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


BASE_DIR = './run'
GRAPH_RECORD_DIR = './run/comp_graph_trace.pkl'


def main_matlab(weights, inputs, comp_graph):
  os.makedirs(BASE_DIR, exist_ok=True)
  with open(GRAPH_RECORD_DIR, 'wb') as f:
    pickle.dump((weights, inputs, comp_graph), f)

  y = eval_m(comp_graph, {})
  print(y)

def gen_sample_data(size: int):
  x = np.random.uniform(-1, 1, (size, 1, 1))
  y = np.concatenate([
    x ** 2,
    -x,
  ], axis=1)
  return torch.from_numpy(x), torch.from_numpy(y)

if __name__ == '__main__':
  os.makedirs(BASE_DIR, exist_ok=True)
  with open(GRAPH_RECORD_DIR, 'rb') as f:
    (weights, inputs, comp_graph) = pickle.load(f)
    
  # init weights
  train_weights = {
    k: (w['Uuid'], nn.Parameter(torch.from_numpy(w['Data']))) \
      for k, w in weights.items()
  }
  
  weights_dict = {
    k: w1 for k, (_, w1) in train_weights.items()
  }
  uuid_dict = {
    k: w0 for k, (w0, _) in train_weights.items()
  }
  
  x_uuid = inputs['x']['Uuid']
  
  def forward_func(x_dict, w):
    eval_m_dict = {
      uuid_dict[k]: p for k, p in w.items()
    }
    eval_m_dict[x_uuid] = x_dict['x']
    
    y = eval_m(comp_graph, eval_m_dict)
    return y
  
  
  model = MatlabWrappedModule(
    params=weights_dict, 
    forward=forward_func
  )
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  num_epochs = 200
  num_batches = 16
  batch_size = 128
  for epoch in range(num_epochs):
    data = [gen_sample_data(batch_size) for _ in range(num_batches)]
    for i, (inputs, targets) in enumerate(data):
      # forward
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      # backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
