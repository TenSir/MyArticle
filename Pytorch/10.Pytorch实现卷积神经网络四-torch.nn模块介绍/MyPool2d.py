import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 读取图片,并转换为Numpy数组
myPic = Image.open('cat.jpg')
myPic_N = np.array(myPic.convert('L'), dtype=np.float32)
print('myPic_N Shape:',myPic_N.shape)
# 进行reshape
pich,picw = myPic_N.shape
mypic_n_gray = torch.from_numpy(myPic_N.reshape((1,1,pich,picw)))
print('mypic_n_gray shape:',mypic_n_gray.shape)

# 核大小
kersize = 7
ker = torch.ones(kersize,kersize,dtype=torch.float32)*-1
ker[3,3] = 25
ker = ker.reshape((1,1,kersize,kersize))
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


# 进行池化操作
MyAvgpool2 = nn.AvgPool2d(kernel_size=2,stride=2)
avgpool2_out = MyAvgpool2(imconv2dout)
avgpool2_out_im = avgpool2_out.squeeze()
print('avgpool2_out shape:',avgpool2_out.shape)

# 可视化池化后的结果
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(avgpool2_out_im[0].data,cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(avgpool2_out_im[1].data,cmap=plt.cm.gray)
plt.axis("off")
plt.show()
