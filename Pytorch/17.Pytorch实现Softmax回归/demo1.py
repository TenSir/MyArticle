import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms



def load_dataset():
    fmnist_train = torchvision.datasets.FashionMNIST(root="F:\DataSet\DLPytorch\Data\FashionMNIST",  train=True, download=True,transform=transforms.ToTensor())
    fmnist_test = torchvision.datasets.FashionMNIST(root="F:\DataSet\DLPytorch\Data\FashionMNIST",   train=False, download=True,transform=transforms.ToTensor())
    return fmnist_train, fmnist_test

# fmnist_train, fmnist_test = load_dataset()
# print(len(fmnist_train))
# print(len(fmnist_test))


def get_fmnist_labels(labels):
    # 获取对应的标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def display_images_1(imgs, num_rows, num_cols, scale=2.0):
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
    # 显示图片
    plt.show()
    return axes

# 这里加载的是整个测试集fmnist_train，所以设置bacth_size获取，并使用iter进行获取
# X, y = next(iter(data.DataLoader(fmnist_train, batch_size=3*9)))
# axes = display_images(X.reshape(3*9, 28, 28), 3, 9)

def display_images(imgs, num_rows, num_cols, scale=3,titles=None,):
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



def load_data_fashion_mnist(batch_size, resize=None):

    root = "F:\DataSet\DLPytorch\Data\FashionMNIST"

    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    fmnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    fmnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    return (data.DataLoader(fmnist_train, batch_size, shuffle=True),
            data.DataLoader(fmnist_test, batch_size, shuffle=False))

# train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
# print(len(train_iter))
# print(len(test_iter))
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break




num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
# print(len(W))
# print(len(b))



def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

# X = torch.tensor([[1,2,3], [4,5,6],[7,8,9]])
# res = softmax(X)
# print(res)
# print(res.sum(1))


def softmaxnet(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
    # return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


def cross_entropy_2(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))



# y = torch.tensor([0, 1, 2])
# y_hat = torch.tensor([
#     [0.1, 0.2, 0.7],
#     [0.2, 0.1, 0.7],
#     [0.4, 0.5, 0.1]
# ])
# print(y_hat[[0, 1, 2], y])

# print(cross_entropy(y_hat,y).shape)
# print(cross_entropy_2(y_hat,y).shape)



def accuracy(y_hat,y):
    print(y_hat.argmax(1))
    acc = (y_hat.argmax(1) == y).float().mean()
    return acc.item()

y = torch.tensor([0, 1, 2])
y_hat = torch.tensor([
    [0.1, 0.2, 0.7],
    [0.2, 0.1, 0.7],
    [0.4, 0.5, 0.1]
])

# y = torch.tensor([[2,1]])
# y_hat = torch.tensor([[0.1,0.3,0.6],[0.5,0.4,0.1]])

print(accuracy(y_hat,y))


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# print(accuracy(y_hat,y))

def evaluate_accuracy(test_data_iter, softmaxnet):
    acc_sum, n = 0.0, 0
    for X, y in test_data_iter:
        # 预测
        y_hat = softmaxnet(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy_1(net, data_iter): #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# train_iter, test_iter = load_data_fashion_mnist(256)
# print(evaluate_accuracy(softmaxnet,test_iter))


def mysgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def my_softmax_train(softmaxnet, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    # 循环数
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = softmaxnet(X)
            l = cross_entropy(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            # 反向传播优化
            l.backward()
            if optimizer is None:
                mysgd(params, lr, batch_size)
            else:
                optimizer.step()
            # 统计准确率
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, softmaxnet)
        print(f"epoch {epoch + 1}, loss {(train_l_sum / n)}, train acc {train_acc_sum / n}, test acc {test_acc}")

batch_size = 256
num_epochs = 5
lr = 0.1

train_iter, test_iter = load_data_fashion_mnist(256)
my_softmax_train(softmaxnet, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


# 取一次测试集的数据
def predict_test(softmaxnet, test_iter, n=9):
    for X, y in test_iter:
        break
    trues = get_fmnist_labels(y)
    preds = get_fmnist_labels(softmaxnet(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    display_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_test(softmaxnet,test_iter)