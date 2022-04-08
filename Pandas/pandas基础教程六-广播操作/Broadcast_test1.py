import pandas as pd

a_1 = pd.Series([0,1])
b_1 = pd.DataFrame([[0,0,0],[10,10,10],[20,20,20],[30,30,30]],
                 index=['A','B','C','D'])
# print(a)
# print(b)
# print(a + b)
# print(a_1 + b_1)

# c = pd.Series([0.1,0.2,0.3])
# d = 10
# print(c*d)
# print(c/10)



# import numpy as np
#
# A = np.array([
#     [2,2,3],
#     [1,2,3]
# ])
# B = np.array([
#     [1,1,3],
#     [2,2,4]
# ])
#
# print(A + B)
# print(A * B)


# import numpy as np
#
# E  = np.array([
#     [0, 0, 0],
#     [1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3]
# ])
#
# F = np.array([1, 2, 3])
# sum = E + F
# print(E.shape)
# print(F.shape)
# print(sum.shape)
# print(sum)
#


#
# import numpy as np
# E = np.array([
#     [0, 0, 0],
#     [1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3]
# ])
# F_1 = np.array([
#     [1],
#     [2],
#     [3],
#     [4]
# ])
#
# sum2 = E + F_1
# print(sum2)


# import torch
# a = torch.arange(0,6).reshape((6,))
# b = torch.arange(0,12).reshape((2,6))
# res = torch.mul(a,b)
# print(a.shape)
# print(b.shape)
# print(res.shape)
# print(res)


import torch
x=torch.empty((0,))
y=torch.empty(2,2)

print(x)
print(y)
print(x.shape)
print(y.shape)

print(x + y)
