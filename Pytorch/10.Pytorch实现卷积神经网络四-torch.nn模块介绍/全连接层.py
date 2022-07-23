import torch

# 输入维度[128，20]
x = torch.randn(128, 20)
nn_linear = torch.nn.Linear(20, 30)
output = nn_linear(x)

print('m.weight.shape:', nn_linear.weight.shape)
print('m.bias.shape:', nn_linear.bias.shape)
print('output.shape:', output.shape)