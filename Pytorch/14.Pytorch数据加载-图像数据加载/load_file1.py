import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10,FashionMNIST
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# 创建训练数据集
trainData = CIFAR10(
    root = "./data/CIFAR10",
    train = False,
    transform = transforms.ToTensor(),
    # download = False
    download = True
)

# 数据加载器
trainLoader = Data.DataLoader(
    dataset=trainData,
    batch_size=64,
    shuffle=True,
)

print('batch 个数为：',len(trainLoader))

# 数据显示
for step ,(X,Y) in enumerate(trainLoader):
    if step < 1:
        imgs = torchvision.utils.make_grid(X,padding=0)
        print(imgs.shape) # torch.Size([3, 256, 256])
        imgs = np.transpose(imgs,(1,2,0))
        print(imgs,'\n')
        plt.imshow(imgs)
        plt.show()