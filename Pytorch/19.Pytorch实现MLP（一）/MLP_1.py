import os
import sys
print(sys.path)
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms


def load_data_fashion_mnist(batch_size, resize=None):

    root = "E:\DataSet\DLPytorch\Data\FashionMNIST"

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    fmnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    fmnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    return (data.DataLoader(fmnist_train, batch_size, shuffle=True),
            data.DataLoader(fmnist_test, batch_size, shuffle=False))




import torch.nn as nn
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]


def my_relu(X):
    # 初始化与X形状一致的张量tensor a
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    # 更改形状，将X转换为自适应的批量大小，长度为num_imputs的向量
    X = X.reshape((-1, num_inputs))
    # 这⾥“@”代表矩阵乘法，注意满足矩阵乘法的规则，W1维度(784,256),b1为256
    H = my_relu(X@W1 + b1)
    # print("H的维度：",H.shape)
    return (H@W2 + b2)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    # 进行累加
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        # 寻找哪一项是最大的
        y_hat = y_hat.argmax(axis =1)
    cmp = y_hat.type(y.dtype) == y
    # print('CMP:',cmp)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    # 将模型设置为评估模式
    if isinstance(net, torch.nn.Module):
        net.eval()
    # 预测正确的数量和预测数量
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    # 精度
    return metric[0] / metric[1]

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
optimizer = torch.optim.SGD(params, lr=lr)

def train(net, train_data, loss, optimizer):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 累加训练损失值、累加训练精度，统计样本数
    # 初始化
    metric = Accumulator(3)
    for X, y in train_data:
        # 计算梯度和更新参数
        y_hat = net(X)
        # 样本损失
        l = loss(y_hat, y)
        if isinstance(optimizer, torch.optim.Optimizer):
            # 梯度清零，即将优化器的梯度清零，避免梯度累加的影响
            optimizer.zero_grad()
            # 利用反向传播算法计算损失函数关于模型参数的梯度。
            l.mean().backward()
            # l.sum().backward()
            # 利用优化器对模型参数进行更新，以最小化损失函数。
            optimizer.step()
        else:
            l.sum().backward()
            optimizer(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 加载数据
train_data,test_data = load_data_fashion_mnist(256)

for epoch in range(num_epochs):
        train_loss, train_acc = train(net, train_data, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_data)
        print("epoch", epoch + 1, ', train_loss', "{:.3f}".format(train_loss), ', train_acc', "{:.3f}".format(train_acc), ', test_acc', "{:.3f}".format(test_acc))














