import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# 设置变换操作集合
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(100),
    transforms.ToTensor()
])

data_dir = "./temPic/"
mydata = ImageFolder(data_dir,
                     transform=train_data_transforms)

mydata_load = DataLoader(
    mydata,
    batch_size=2,
    shuffle=True,
    # num_workers=1
)

# 获取标签等信息
print(len(mydata))
print(mydata.targets)
print(len(mydata_load))

# 获取batch数据
for step,(X,Y) in enumerate(mydata_load):
    if step < 1:
        print(X.shape)
        print(Y.shape)
        print(X[0].shape)
        # print(X[0].shape)
        # print(X[0],'\n')
        im = np.array(X[0])
        # print(im)
        # #上一步图像格式为[通道数，长，宽],需要使用transpose实现转置，格式变为[长，宽，通道数]
        img = np.transpose(im, (1, 2, 0))
        plt.imshow(img)
        plt.show()