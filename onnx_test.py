import torch
import torch.nn as nn
import torch.onnx

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = SimpleModel()

# 创建一个输入张量
dummy_input = torch.randn(1, 10)

# 导出模型为 ONNX 格式
torch.onnx.export(model, dummy_input, "simple_model.onnx", input_names=['input'], output_names=['output'])