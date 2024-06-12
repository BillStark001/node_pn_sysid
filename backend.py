import numpy as np

import torch
import torch.nn as nn

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



def main_matlab(weights, forward_func):
  model = MatlabWrappedModule(weights, forward_func)
  traced_model = torch.jit.trace(model, torch.from_numpy(np.array([[1]])))
  traced_model.save('./test_model.pt')


# model = SimpleFCN()

# example_input = torch.randn(1, 10)
# traced_model = torch.jit.trace(model, example_input)

# example_input = torch.randn(1, 10)
# output = traced_model(example_input)
# print(output)