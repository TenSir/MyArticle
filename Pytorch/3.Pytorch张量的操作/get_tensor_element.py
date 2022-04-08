import torch
mytensor = torch.arange(36).reshape(3,3,4)
# print(mytensor)


# print(mytensor[0])
# print(mytensor[1])
# print(mytensor[2])


# 获取第二个维度张量的前两行元素
# print(mytensor[1][0:2])
# print('______________')
# print(mytensor[1,0:2])
# print('______________')
# print(mytensor[1,0:2,:])


# print(mytensor[-1])
# print(mytensor[1][-1][-3:-1])
# print('---------------------------------')
# t1 = torch.arange(12).reshape(3,4)
# t2 = torch.arange(12).reshape(3,4) + 100
# print(t1)
# print(t2)
# print(t1[t1>2])
#
# print(torch.where(t1>6,t1,t2))
# print(torch.where(t1>6,t1,0))


# a = torch.arange(24).reshape(4,6)
# b1 = torch.tril(a,diagonal=0)
# b2 = torch.tril(a,diagonal=2)
# b3 = torch.tril(a,diagonal=-1)
# print(a)
# print(b1)
# print(b2)
# print(b3)



a = torch.arange(12).reshape(3,4)
b = torch.rand(3,4)
c= torch.cat((a,b),dim=0)
d= torch.cat((a,b),dim=1)
# print(c)
# print(d)

# e = torch.stack((a,b),dim=0)
# f = torch.stack((a,b),dim=1)
# print(e)
# print(f)

a = torch.arange(12).reshape(3,4)
# print(torch.chunk(a,chunks=2,dim = 1))

res1,res2,res3 = torch.split(a,(1,1,1),dim = 0)
# print(res1)
# print(res2)
# print(res3)


print(torch.tensor_split(a, 3))
print(torch.tensor_split(a, 3, dim=1))
# print(torch.tensor_split(a, (2,1), dim=1))