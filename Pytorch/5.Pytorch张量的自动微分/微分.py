# import numpy as np
# def grad_myself(func, x, delta=1e-7):
#     fc_1, fc_2 = func(x + delta), func(x - delta)
#     g = (fc_1 - fc_2) / (delta)
#     return g
#
# f = lambda x: x**2 + x**(1/3)
# print(grad_myself(func=f,x=3))


# import torch
# from torch.autograd import Variable
# X = Variable(torch.ones(3,3), requires_grad = True)
# X_1 = torch.ones(3,3)
# print(X)
# print(X_1)

# Y = X + 0.5
## print(Y)
## print(Y.data)

# Z = Y * Y
# print(type(Z))
#
# Z.backward(X)
# print(Z.grad)
# print(Y.grad)
# print(X.grad)


import torch
from torch.autograd import Variable
X = Variable(torch.tensor([3.0]), requires_grad = True)
Y = X ** 2 + X **(1/3)
Y.backward()

print(Y.grad)
print(X.grad)


x = torch.tensor(5.0, requires_grad=True)
y = x.exp()
y.retain_grad()
y.backward()
print(y.grad)

