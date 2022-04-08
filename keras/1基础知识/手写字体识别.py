# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 手写字体识别.py
# @Time    : 2020/8/13 22:33
# @Cnblogs ：Python知识学堂

# 加载keras包含的mnist的数据集
from keras.datasets import mnist
import matplotlib.pyplot as plt
path = r'C:\Users\LEGION\Desktop\keras内置datasets\mnist.npz'

(X_train, y_train), (X_test, y_test) = mnist.load_data(path)
print(len(X_train))

for each in range(4):
    plt.subplot(2,2,each+1)
    plt.imshow(X_train[each], cmap=plt.get_cmap('gray'), interpolation='none')
    plt.title("Class {}".format(y_train[each]))
plt.show()



"""
import numpy as np
from  keras.datasets import mnist
def load_data(path):
    numpy_load= np.load(path)
    X_train, y_train = numpy_load['x_train'], numpy_load['y_train']
    X_test, y_test = numpy_load['x_test'], numpy_load['y_test']
    numpy_load.close()

    return (X_train, y_train), (X_test, y_test)

path = r'C:\\Users\LEGION\Desktop\datasets\mnist.npz'
(X_train, y_train), (X_test, y_test) = load_data(path)
print(X_train.shape)
print(y_train.shape)

"""


