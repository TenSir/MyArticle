import numpy as np
import torchvision
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



class MyCIFAR10(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 ):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.labels = [], []

        def myload(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        if self.train:
            for i in range(1, 6):
                file = root + "\\data_batch_" + str(i)
                data = myload(file)
                for item, label in zip(data[b'data'], data[b'labels']):
                    self.data.append(item)
                    self.labels.append(label)
        else:
            file = root + "\\test_batch"
            data = myload(file)
            for item, label in zip(data[b'data'], data[b'labels']):
                self.data.append(item)
                self.labels.append(label)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        image, label = np.array(image), np.array(label)
        return image, label

    def __len__(self):
        return len(self.data)



# MyCIFAR10
mycifar10 = MyCIFAR10(
    '.\data\CIFAR10\cifar-10-batches-py',
    train=True,
    transform=transforms.ToTensor()
)

# 数据加载器
trainLoader = DataLoader(
    dataset=mycifar10,
    batch_size=64,
    shuffle=True,
)


image, label = mycifar10.__getitem__(1)
img = image.reshape(3, 32, 32).astype('float32')
imgPic = np.transpose(img, (1, 2, 0))/255
plt.imshow(imgPic)
plt.show()
print('batch 个数为：',len(trainLoader))
print('长度：',len(image))