import torch
from torch.autograd import Variable

x = Variable(torch.linspace(-1,1,100).type(torch.FloatTensor))
x = torch.unsqueeze(x,dim=1)
y = 3.14*x + 0.8* torch.rand(x.size())

# 1.定义超参数
epoch = 10000
learning_rate = 0.001

# 2.定义模型
# 输入输出特征均为1维，该函数会初始化权重信息
model = torch.nn.Linear(in_features=1, out_features=1)

# 3.定义损失函数
# 使用均方误差作为损失函数，size_average = False表示使用总误差
criterion= torch.nn.MSELoss(size_average= False)

# 4.定义优化函数
# model.parameters()会自动提取模型中的参数
optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)

# 5.开始训练
for i in range(epoch):
    # 预测
    y_predictions = model(x)
    # 损失计算
    loss = criterion(y_predictions, y)
    if i % 500 ==0:
       print(loss)
    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()


for name, param in model.named_parameters():
    print(name,param)
    print('_________')


# Test Model
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data)