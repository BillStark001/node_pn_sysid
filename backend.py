import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import time

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
MODEL_TRACE_DIR = './run/model_trace.pt'
USE_MODEL_TRACE = False


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
  
  params = {
    k: w1 for k, (_, w1) in train_weights.items()
  }
  param_uuids = {
    k: w0 for k, (w0, _) in train_weights.items()
  }
  input_uuids = {
    k: w['Uuid'] for k, w in inputs.items()
  }
  
  # load traced model if necessary
  traced_model = None
  needs_trace = False
  if USE_MODEL_TRACE:
    if os.path.exists(MODEL_TRACE_DIR):
      traced_model = torch.jit.load(MODEL_TRACE_DIR)
    else:
      needs_trace = True
  
  model = MatlabWrappedModule(
    comp_graph,
    params, 
    param_uuids,
    input_uuids,
    traced_model=traced_model,
  )
  
  # trace
  if needs_trace:
    traced_model_orig = model.traced_model
    x, _ = gen_sample_data(1)
    inputs = tuple(model.get_traced_module_input(x))
    traced_model = torch.jit.trace(traced_model_orig, inputs)
    traced_model.save(MODEL_TRACE_DIR)
  
  # init training
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=300, factor=0.5)

  num_epochs = 1000
  num_batches = 16
  batch_size = 128
  
  tstart = time.time()
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
      scheduler.step(loss)
      
    if epoch % 50 == 49:
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
  tend = time.time()
  print(f'Elapsed time: {tend - tstart:.3f}s')
