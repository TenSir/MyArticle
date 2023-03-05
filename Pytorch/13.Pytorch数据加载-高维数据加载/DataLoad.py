# import torch
# import numpy as np
# import torch.utils.data as Data
# from sklearn.datasets import load_boston
#
# import warnings
# warnings.filterwarnings("ignore")
#
# X,Y = load_boston(return_X_y=True)
# print(X.dtype)
# print(Y.dtype)
# # 将数组转换为Tensor
# train_x = torch.from_numpy(X.astype(np.float32))
# train_y = torch.from_numpy(Y.astype(np.float32))
# train_data = Data.TensorDataset(train_x,train_y)
# # 定义数据加载器
# Loader = Data.DataLoader(
#     dataset=train_data,
#     batch_size=32,
#     shuffle=True
# )

# for inx,(x,y) in enumerate(Loader):
#     print('inx:',inx)
#     print('x:',x)
#     print('y:',y)
#     break




import torch
import numpy as np
import torch.utils.data as Data
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings("ignore")

X,Y = load_iris(return_X_y=True)
print(X.dtype)  # float64
print(Y.dtype)  # int32

# 将数组转换为Tensor
train_x = torch.from_numpy(X.astype(np.float32))
train_y = torch.from_numpy(Y.astype(np.int64))
train_data = Data.TensorDataset(train_x,train_y)
# 定义数据加载器
Loader = Data.DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True
)

for inx,(x,y) in enumerate(Loader):
    print('inx:',inx)
    print('x:',x)
    print('y:',y)
    break