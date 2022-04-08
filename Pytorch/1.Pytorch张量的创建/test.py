import torch
a = torch.Tensor(3,4)
print(a)
print(a.type())

print('_______________________________')
b1 = torch.ones(2,3)
b2 = torch.ones(2,3,dtype=torch.int)
print(b1)
print(b1.type())
print(b2)
print(b2.type())

print('_______________________________')
c = torch.zeros(3,4)
print(c)

print('_______________________________')
d = torch.rand(3,4)
print(d)

print('_______________________________')
e = torch.randn(3,4)
print(e)

print('_______________________________')
f = torch.eye(3)
print(f)


print('_______________________________')
g = torch.arange(1,10,2)
print(g)

print('_______________________________')
h = torch.linspace(1,10,5)
print(h)

print('_______________________________')
i =  torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
print(i)

print('_______________________________')
j =torch.Tensor(2,3).uniform_(-1,1)
print(j)

print('_______________________________')
h = torch.randperm(5)
print(h)

print('_______________________________')
import numpy as np
array = np.array([1, 2, 3, 4, 5])
tensor_array = torch.from_numpy(array)
print(tensor_array)
print('_______________________________')



import torch
one=torch.Tensor([1,2])
two=torch.tensor([1,2])

print(one)
print(two)
print(one.type())
print(two.type())


