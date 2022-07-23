import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 读取图片,并转换为Numpy数组
myPic = Image.open('cat.jpg')
myPic_N = np.array(myPic.convert('L'), dtype=np.float32)

# 1----------------------------------------------------------------
# plt.figure(figsize=(6, 6))
# plt.imshow(myPic_N, cmap=plt.cm.gray)
# plt.axis("off")
# plt.show()
# print(myPic_N.dtype)
# print(myPic_N.shape)


## 将数组转化为张量
pich,picw = myPic_N.shape
mypic_n_gray = torch.from_numpy(myPic_N.reshape((1,1,pich,picw)))
print(mypic_n_gray.shape)


# 2----------------------------------------------------------------
# 定义卷积并进行卷积操作
# kersize = 7
# # 数值全部是-1
# ker = torch.ones(kersize,kersize,dtype=torch.float32)*-1
# # (3,3)的位置为25
# ker[3,3] = 25
# ker = ker.reshape((1,1,kersize,kersize))
# 设置in_channels和out_channels均为1，进行卷积操作
# conv2d = nn.Conv2d(1,1,(kersize,kersize),bias = False)
# # 设置卷积时使用的核
# conv2d.weight.data = ker
# ## 对灰度图像进行卷积操作
# imconv2dout = conv2d(mypic_n_gray)
# print('imconv2dout shape:',imconv2dout.shape)
# # ## 对卷积后的输出图像进行复原
# imconv2dout_pic = imconv2dout.data.squeeze()
# print('imconv2dout_pic shape:',imconv2dout_pic.shape)
# print("核：",ker)
# #可视化卷积后的图像
# plt.figure(figsize=(8, 8))
# plt.imshow(imconv2dout_pic,cmap=plt.cm.gray)
# plt.axis("off")
# plt.show()

# 核大小
kersize = 7
ker = torch.ones(kersize,kersize,dtype=torch.float32)*-1
ker[3,3] = 25
ker = ker.reshape((1,1,kersize,kersize))
print("核：",ker)
# 设置输出的通道数为2，进行卷积操作
conv2d = nn.Conv2d(in_channels=1,out_channels=2,kernel_size=(kersize,kersize),bias = False)
# 设置卷积时使用的核,第一个核使用边缘检测核
print('conv2d.weight.data shape:',conv2d.weight.data.shape)
conv2d.weight.data[0] = ker
# 对灰度图像进行卷积操作
imconv2dout = conv2d(mypic_n_gray)
print('imconv2dout shape:',imconv2dout.shape)
# 对卷积后的输出图像进行复原
imconv2dout_pic = imconv2dout.data.squeeze()
print('imconv2dout_pic shape:',imconv2dout_pic.shape)
# 可视化
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(imconv2dout_pic[0],cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(imconv2dout_pic[1],cmap=plt.cm.gray)
plt.axis("off")
plt.show()

print('conv2d.weight.data shape:',conv2d.weight.data)