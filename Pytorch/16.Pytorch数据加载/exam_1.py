import torch

# x = torch.arange(4.0)  # 0,1,2,3
# x.requires_grad_(True)
# y = 2 * torch.dot(x, x)
# # print(y)
# y.backward()  # 4倍的X,即4x
# print(x.grad)  # tensor([ 0.,  4.,  8., 12.])


# 另一个函数
# x.grad.zero_()
# print(x)  # tensor([0., 1., 2., 3.], requires_grad=True)
# y = x.sum()
# print(y)  # tensor(6., grad_fn=<SumBackward0>)
# y.backward()
# print(x.grad)  # tensor([1., 1., 1., 1.])



def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)