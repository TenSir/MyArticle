# import pandas as pd
# import torch
# class MyIterableDataset(torch.utils.data.IterableDataset):
#
#     def __init__(self, myfile):
#         self.data = pd.read_csv(myfile, iterator=True, header=None, chunksize=100)
#
#     def __iter__(self):
#             # 遍历读取
#             for eachdata in self.data:
#                 yield eachdata
#
# dataset = MyIterableDataset('.\\titanic\\train.csv')
# for data in dataset:
#     print(data)


# import torch
# from torch.utils.data import TensorDataset, DataLoader
#
# X = torch.tensor([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9],
#                   [11, 22, 33],
#                   [44, 55, 66],
#                   [77, 88, 99],
#                   [111, 222, 333],
#                   [444, 555, 666],
#                   [777, 888, 999],
#                   [1111, 2222, 3333],
#                   [4444, 5555, 6666],
#                   [7777, 8888, 9999]])
#
# Y = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11])
#
# mydata = TensorDataset(X, Y)
# for x_train, y_label in mydata:
#     print(x_train, y_label)
#
# # 使用DataLoader进行加载
# loaderData = DataLoader(dataset=mydata, batch_size=5, shuffle=True)
# for inx, data in enumerate(loaderData, 1):
# # 输出数据和batch详情
#     dat, label = data
#     print(f'batch:{inx} x:{dat}  y: {label}')




# import torch
# from torch.utils.data import Sampler
# from torch.utils.data import SequentialSampler
#
# mydata = [12,14,15,17,2,0]
# mydata_seq = SequentialSampler(mydata)
# for index in mydata_seq:
#     print("index: {}, data: {}".format(str(index), str(mydata[index])))


# from torch.utils.data import RandomSampler
# mydata = [12,14,15,17,2,0]
# samplerdat = RandomSampler(data_source=mydata, replacement=True)
# for index in samplerdat:
#     print("index: {}, data: {}".format(str(index), str(mydata[index])))



import torch
from torch.utils.data import TensorDataset, DataLoader

def my_test():
    X = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [11, 22, 33],
                      [44, 55, 66],
                      [77, 88, 99],
                      [111, 222, 333],
                      [444, 555, 666],
                      [777, 888, 999],
                      [1111, 2222, 3333],
                      [4444, 5555, 6666],
                      [7777, 8888, 9999]])

    Y = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11])

    mydata = TensorDataset(X, Y)

    # for x_train, y_label in mydata:
    #     print(x_train, y_label)

    # 构建数据加载器
    loader = DataLoader(
        dataset=mydata,
        batch_size=5,
        shuffle=True,
        num_workers=2
    )
    # 假设我们在训练时，进行2次epoch
    for epoch in range(2):
        for step, (x_train, y_label) in enumerate(loader):
            print('Epoch:', epoch)
            print('Step:', step)
            print('batch x:',x_train.numpy())
            print('batch y:', y_label.numpy())
        print('——————————————————————————')

if __name__ == '__main__':
    my_test()