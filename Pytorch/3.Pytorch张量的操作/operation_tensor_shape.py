import torch



a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float)
b = torch.ones_like(a)
# print(a)
# print(b)
# print(a.dtype)

# print(a.shape)
# print(a.shape[0])
# print(a.resize(1,9))
# print(a.resize(9,1))
# print(a.resize_(9,1))
# print(a)
# print(a.numpy())


# c = torch.reshape(a,shape=(1,9))
# print(c)
# print(a)


# a.resize_as(b)

# print(a.view(1,9))
# print(a.view(3,3))
# print(a.permute(1,0))
# print(a.permute(3,0,1))

# import numpy as np
# t = np.arange(18)
# t = np.reshape(t, (3,3,2))
# t = torch.from_numpy(t)
# print(t)
# print(torch.flatten(t))
# print(torch.flatten(t,start_dim=0, end_dim=1))
# print(torch.flatten(t,start_dim=1, end_dim=2))
#
# d1 = torch.unsqueeze(input=a,dim=0)
# print(d1)
# print(d1.size())
# print(d1.dim())

import numpy as np
t = np.arange(8)
t = np.reshape(t, (4,1,2))
t = torch.from_numpy(t)
print(t)
print(torch.squeeze(input=t,dim=1))
print(torch.squeeze(input=t,dim=0))
