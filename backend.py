import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import importlib
import syntax_tree
importlib.reload(syntax_tree)

from syntax_tree import MatlabTracedModule, MatlabWrappedModule, eval_m
from utils import USE_CACHE, cache
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


BASE_DIR = './run'
GRAPH_RECORD_DIR = './run/comp_graph_trace.pkl'
MODEL_TRACE_DIR = './run/model_trace.pt'
USE_MODEL_TRACE = True


@cache('./run/main_matlab_cache.pkl')
def main_matlab(weights, inputs, comp_graph):
  os.makedirs(BASE_DIR, exist_ok=True)
  with open(GRAPH_RECORD_DIR, 'wb') as f:
    pickle.dump((weights, inputs, comp_graph), f)

  y = eval_m(comp_graph, {})
  print(y)
  return y

def gen_sample_data(size: int):
  x = np.random.uniform(-1, 1, (size, 1, 1))
  y = np.concatenate([
    x ** 2,
    -x,
  ], axis=1)
  return torch.from_numpy(x), torch.from_numpy(y)

if __name__ == '__main__':
  main_matlab(USE_CACHE)
  exit(1)
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
