import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms


def get_fmnist_labels(labels):
    # 获取对应的标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


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


num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def softmaxnet(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
    # return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


def evaluate_accuracy(test_data_iter, softmaxnet):
    acc_sum, n = 0.0, 0
    for X, y in test_data_iter:
        # 预测
        y_hat = softmaxnet(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


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

# 参数设置
batch_size = 256
num_epochs = 5
lr = 0.1

# 加载数据
train_iter, test_iter = load_data_fashion_mnist(256)
# 模型训练
my_softmax_train(softmaxnet, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 取一次测试集的数据，进行测试
def predict_test(softmaxnet, test_iter, n=9):
    for X, y in test_iter:
        break
    trues = get_fmnist_labels(y)
    preds = get_fmnist_labels(softmaxnet(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    display_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_test(softmaxnet,test_iter)