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

model = SimpleFCN()

example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)

example_input = torch.randn(1, 10)
output = traced_model(example_input)
print(output)