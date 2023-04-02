import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms




def load_data_fashion_mnist(batch_size=None, resize=None):

    root = "E:\DataSet\DLPytorch\Data\FashionMNIST"

    if batch_size is None:
        batch_size = 256
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    fmnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    fmnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    return (data.DataLoader(fmnist_train, batch_size, shuffle=True),
            data.DataLoader(fmnist_test, batch_size, shuffle=False))



in_features = 28 * 28
out_features = 10

# 定义模型， 全连接层
class MySoftmaxNet(nn.Module):
    def __init__(self, in_features, out_features):
        # 子类继承了父类的所有属性和方法，父类属性使用父类方法来进行初始化
        # super(MySoftmaxNet,self).__init__()
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    # 向前传播
    def forward(self, x):
        # x.shape: torch.Size([batch, 1, 28, 28])
        # print(x.shape)
        return self.linear(x.view(x.shape[0], -1))

# 定义模型
# net = MySoftmaxNet(in_features, out_features)
# 初始化模型参数
# torch.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
# torch.nn.init.zeros_(net.linear.bias)


# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net = nn.Sequential(nn.Flatten(), nn.Linear(in_features, out_features))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)
init_weights(net)
net.apply(init_weights);



# 损失函数定义
loss = nn.CrossEntropyLoss(reduction='none')

# 优化器
optimizer  = torch.optim.SGD(net.parameters(), lr=0.1)



# 计算分类准确性
def cal_accurary(data_iter, net):
    acc_sum,n = 0.0,0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # 计算准确判断的数量,shape[0]即零维度（列）的元素数量
        n += y.shape[0]
    return acc_sum / n


# 定义训练函数
# 学习周期
num_epochs = 1
# num_epochs = 5
def train(net, train_iter, test_iter, loss, num_epochs, batch_size, lr, optimizer):
    # 损失值、正确数量、总数
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    n = 0
    for epoch in range(1, num_epochs + 1):
        for X, y in train_iter:
            y_hat = net(X)
            # 数据集损失函数的值等于每个样本的损失函数值的和。
            l = loss(y_hat, y).sum()
            # 梯度清零
            optimizer.zero_grad()
            # 损失函数梯度计算
            l.backward()
            optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        # 测试集准确度
        test_acc = cal_accurary(test_iter, net)
        print('epoch %d, loss= %.4f, train_acc=%.4f, test_acc=%.4f' % (epoch, train_loss_sum / n, train_acc_sum / n, test_acc))





lr = 0.1
batch_size = 256

train_iter,test_iter = load_data_fashion_mnist(batch_size = 256)
train(net, train_iter, test_iter, loss, num_epochs, batch_size, lr, optimizer)


# 获取数据正确的标签，即将样本的类别数字转换成文本
def get_fmnist_labels(labels):
    # 获取对应的标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]



#显示图像
def display_images_1(imgs, num_rows, num_cols,titles=None,):
    scale = 3
    # 设置图像画图大小
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图⽚张量
            ax.imshow(img.numpy())
        else:
            # PIL图⽚
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
        if titles:
            ax.set_title(titles[i])
    # 显示图片
    plt.show()
    return axes


# 绘制图像
from IPython import display
def display_images(images,labels):
    # 绘制矢量图b
    display.display_svg()
    _,figs = plt.subplots(1,len(images),figsize=(12,12))
    #设置添加子图的数量、大小
    for f,img,label in zip(figs,images,labels):
        f.imshow(img.view(28,28).numpy())

        f.axes.get_xaxis().set_visible(True)
        f.axes.get_yaxis().set_visible(True)

        f.set_title(label)
    plt.show()


# 迭代从测试集中获得样本和标签
X, y = iter(test_iter).next()
print(X.shape)
print(y.shape)


# 获取真实的标签
true_labels = get_fmnist_labels(y.numpy())
# 获取预测的标签
pred_labels = get_fmnist_labels(net(X).argmax(dim=1).numpy())
# 将真实标签和预测得到的标签加入到图像上
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

display_images(X[0:9],titles[0:9])