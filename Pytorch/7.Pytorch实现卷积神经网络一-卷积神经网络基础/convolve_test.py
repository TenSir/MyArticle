import numpy as np
from scipy import signal

my_pic = np.array([[1,	0, 1, 0, 0, 1],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1 ,1, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1]
                ])

kernal = np.array([[1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, 1]
                 ])
# 卷积计算
res = signal.convolve2d(my_pic,kernal,mode='valid')
print(res)