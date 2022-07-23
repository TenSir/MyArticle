import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 读取图片,并转换为Numpy数组
myPic = Image.open('cat.jpg')
myPic_N = np.array(myPic.convert('L'), dtype=np.float32)

nn.AvgPool2d()



