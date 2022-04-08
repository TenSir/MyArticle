import sys
import torch
print(torch.get_default_dtype())


tensor_1 = torch.tensor([1,2,3])
print(torch.is_floating_point(tensor_1))
print(tensor_1.dtype)


tensor_2 = torch.tensor([1,2.0,3],dtype=torch.complex64)
print(tensor_2)
print(torch.is_complex(tensor_2))


# from sklearn.model_selection import train_test_split

print("line-3-------------------------------------------")


uint_tensor = torch.ones((2,3), dtype=torch.uint8)
int_tensor = torch.ones((2,3), dtype=torch.int)
float_tensor = torch.ones((2,3), dtype=torch.float)
long_tensor = torch.ones((2,3), dtype=torch.long)
double_tensor = torch.ones((2,3), dtype=torch.double)
complex_float_tensor = torch.ones((2,3), dtype=torch.complex64)
complex_double_tensor = torch.ones((2,3), dtype=torch.complex128)
bool_tensor = torch.ones((2,3), dtype=torch.bool)

long_zerodim = torch.tensor((2,3), dtype=torch.long)
int_zerodim = torch.tensor((2,3), dtype=torch.int)


# 1.int64加上常数结果为默认的int64类型
print(torch.add(5, 5))
print(torch.add(5, 5).dtype)

# 2.torch.int 与常数相加结果为torch.int32类型
print(int_tensor + 10)
print((int_tensor + 10).dtype)

# 3.torch.int与torch.long数据相加为torch.int64类型
print(int_tensor + long_tensor)
print((int_tensor + long_tensor).dtype)

# 4. troch.float与torch.double相加结果为torch.float64类型
print(float_tensor + double_tensor)
print((float_tensor + double_tensor).dtype)

# 5.torch.complex64与torch.complex128相加结果为torch.complex128
print(complex_float_tensor + complex_double_tensor)
print((complex_float_tensor + complex_double_tensor).dtype)

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


# 先检查一下cuda是否可用
print(torch.cuda.is_available())

# 检查可用的cuda的数量
print(torch.cuda.device_count())

# 返回当前cuda使用的设备的索引号
print(torch.cuda.current_device())

# 通过字符串：
print(torch.device('cuda:0'))
print(torch.device('cpu'))
print(torch.device('cuda'))


# 通过字符串和设备序号：
print(torch.device('cuda', 0))
print(torch.device('cpu', 0))


# 在CPU上设置一个tensor
cuda1 = torch.device('cuda:0')
cuda_a_tensor = torch.randn((2,3), device=cuda1)
print(cuda_a_tensor)
print(cuda_a_tensor.dtype)
print(cuda_a_tensor.device)

print('___________________________________________')
# torch.randn((2,3), device=torch.device('cuda:0'))
# torch.randn((2,3), device='cuda:0')
# torch.randn((2,3), device=0)

device_cpu = torch.device("cpu")  # 声明cpu设备
device_cuda = torch.device('cuda')  #设备gpu设备
data = torch.Tensor([1,2,3,4,5])
print(data.device)
data_1 = data.to(device_cpu)
print(data_1.device)
data_2 = data.to(device_cuda)
print(data_2.device)
print('___________________________________________')

data = torch.Tensor([1,2,3,4,5])
print(data.dtype)
print(data.to('cuda'))


x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(x.stride())


