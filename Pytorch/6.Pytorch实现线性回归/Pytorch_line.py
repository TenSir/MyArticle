import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def gen_data(num):
    # 构建num个数据点(X,Y)
    X = Variable(torch.linspace(-1,1,num).type(torch.FloatTensor))
    # Y = 3.14X + 0.8*b
    # 偏差b满足均值为0，方差为取得标准正太分布
    Y = 3.14*X + 0.8 * torch.rand(X.size())
    return X,Y

def show_data(X,Y):
    x_train = X[:-20]
    x_test = X[-20:]
    y_train = Y[:-20]
    y_test = Y[-20:]

    plt.figure(figsize=(10,8))
    # plt.plot(x_train.data.numpy(),y_train.data.numpy(),'o')
    plt.scatter(x_train.data.numpy(),y_train.data.numpy())
    plt.xlabel('x_feature')
    plt.ylabel('y_target')
    plt.show()
    return x_train,y_train,x_test,y_test


X,Y = gen_data(200)
x_train,y_train,x_test,y_test = show_data(X,Y)

# 定义超参数
epoch = 10000
learning_rate = 0.001

# 初始化参数值
W = Variable(torch.rand(1),requires_grad = True)
B = Variable(torch.rand(1),requires_grad = True)

for i in range(epoch):
    # 模型
    y_predictions = W * x_train + B
    # 计算损失
    square_loss = torch.mean((y_predictions - y_train) **2)
    # 反向求导
    square_loss.backward()
    # 根据上一步的计算结果来更新参数值，大家参考梯度下降的公式
    # W.data = W.data - learning_rate * W.grad.data
    # B.data = B.data - learning_rate * B.grad.data
    # 或者使用以下代码进行更新
    W.data.add_(-learning_rate * W.grad.data)
    B.data.add_(-learning_rate * B.grad.data)
    # 打印loss信息
    if i % 5000 == 0:
        print(square_loss)
    # 清空自动求导保存的梯度信息，避免在backend的过程中反复的进行累计
    W.grad.data.zero_()
    B.grad.data.zero_()

express = str(W.data.numpy()[0]) + 'X + ' + str(B.data.numpy()[0])
print('W:',W)
print('B:',B)
print(express)

plt.figure(figsize=(10,6))
plt.scatter(x_train.data.numpy(),y_train.data.numpy())
plt.plot(x_train.data.numpy(),y_predictions.data.numpy(),'r-',lw = 6)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


y_pred = W * x_test + B
print('x_test 预测结果:',y_pred)
print('真实的y_test:',y_test)