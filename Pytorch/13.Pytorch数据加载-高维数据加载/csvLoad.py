import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatasetFromCSV(Dataset):
    def __init__(self, datapath):
        # 读文件夹下每个数据文件的名称
        self.allfilename = os.listdir(datapath)
        self.alldatapath = []

        for eachfilename in self.allfilename:
            self.alldatapath.append(os.path.join(datapath,eachfilename))

    def __len__(self):
        # 目录下有多少个文件
        return len(self.allfilename)

    def __getitem__(self, index):
        # 读取每一个数据
        data = pd.read_csv(self.alldatapath[index])
        # 张量形式
        data = torch.tensor(data.values)
        return data


# in_dir = r'F:\MyArticle\Pytorch\13.Pytorch数据加载-高维数据加载\titanic'
# 读取数据集
# train_dataset = DatasetFromCSV(datapath=in_dir)
# print(train_dataset.__len__())
# # 加载数据集
# train_iter = DataLoader(train_dataset)



class MyDataset(Dataset):

    def __init__(self, filepath):
        # 获取数据
        trainData = pd.read_csv(filepath)
        # 转换为Tensor的形式
        self.feature = torch.from_numpy(trainData.iloc[:,1:-1].values)
        self.target = torch.from_numpy(trainData.iloc[:,[-1]].values)

        # 获取数据的大小
        self.len = trainData.shape[0]

    def __getitem__(self, index):
        # 获取第index条数据的特征及标签
        return self.feature[index], self.target[index]

    def __len__(self):
        # 数据长度
        return self.len


if __name__ == "__main__":
    dataset = MyDataset('mydata.csv')
    print(len(dataset))
    print(dataset[2])