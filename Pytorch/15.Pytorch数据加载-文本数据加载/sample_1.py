# from torch.utils.data import Dataset,DataLoader
# import pandas as pd
#
# class MyDataset(Dataset):
#     def __init__(self,data_path):
#         self.data_path = data_path
#         self.df = pd.read_csv(self.data_path, sep='\t', header=None)
#         new_col = ['label', 'sms']
#         self.df.columns = new_col
#
#     def __getitem__(self, index):
#         item = self.df.iloc[index,:]
#         return item.values[0],item.values[1]
#
#     def __len__(self):
#         return self.df.shape[0]
#
# data_path = r"SMSSpamCollection"
# d = CifarDataset(data_path)
# for idx in range(len(d)):
#     # print(idx,d.__getitem__(idx))
#     print(idx, d[idx])


from torch.utils.data import Dataset,DataLoader
import pandas as pd

class MyDataset(Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path, sep='\t', header=None)
        new_col = ['label', 'sms']
        self.df.columns = new_col

    def __getitem__(self, index):
        item = self.df.iloc[index,:]
        return item.values[0],item.values[1]

    def __len__(self):
        return self.df.shape[0]

data_path = r"SMSSpamCollection"
mydataset = MyDataset(data_path)
data_loader = DataLoader(dataset=mydataset, batch_size=20, shuffle=True)

print("总数据长度:",len(mydataset))

for idx,(label,sms) in enumerate(data_loader):
    if idx < 1:
        print("标签:",label)
        print("数据:",sms)
        print("本次批数据对象:",data_loader)
        print("本次批数据长度:",len(data_loader))
        print('________________________________')
