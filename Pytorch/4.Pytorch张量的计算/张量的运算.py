import torch

# torch.add(input, other, *, alpha=1, out=None) # 相加
# torch.sub(input, other, out=None) # 相减
# torch.mul(input, other, out=None) # 相乘
# torch.div(input, other, out=None) # 相除


t1 = torch.ones(2, 5)
t2 = torch.arange(10).reshape(2,5)
# print(t1)
# print(t2)


# add_1 = torch.add(t1,t2)
# add_2 = torch.add(t1,t2,alpha=2)
# print(add_1)
# print(add_2)

# print(t1+t2)
# print(t1+2*t2)

#########################################################
# mul_1 = torch.mul(t1+1,t2)
# t3 = torch.randn(4, 1)
# t4 = torch.randn(1, 4)
# mul_2 = torch.mul(t3,t4)
# print(mul_1)
# print(t3)
# print(t4)
# print(mul_2)

##########################################################
# a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
#                   [ 0.1815, -1.0111,  0.9805, -1.5923],
#                    [ 0.1062,  1.4581,  0.7759, -1.2344],
#                  [-0.1830, -0.0313,  1.1908, -1.4757]])
#
# b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
#
# div_1 = torch.div(a,2)
# div_2 = torch.div(a, b, rounding_mode='trunc')
# div_3 = torch.div(a, b, rounding_mode='floor')
#
# print(div_1)
# print(div_2)
# print(div_3)


t_eq_1 = torch.tensor([1, 2, 3])
t_eq_2 = torch.tensor([1, 1, 3])

print(torch.eq(t_eq_1,t_eq_2))
print(t_eq_1.equal(t_eq_2))
print(torch.equal(t_eq_1, t_eq_2))


