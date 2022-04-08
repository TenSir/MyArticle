# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 3.py
# @Time    : 2020/9/12 11:33
# @Cnblogs ：Python知识学堂

import h5py

# 模型地址
MODEL_PATH = r'F:\kerasdataset\mnist_test.h5'

# 获取每一层的连接权重及偏重
print("读取模型中...")
with h5py.File(MODEL_PATH, 'r') as f:
    dense_1 = f['/model_weights/dense_1/dense_1']
    dense_1_bias =  dense_1['bias:0'][:]
    dense_1_kernel = dense_1['kernel:0'][:]

    dense_2 = f['/model_weights/dense_2/dense_2']
    dense_2_bias = dense_2['bias:0'][:]
    dense_2_kernel = dense_2['kernel:0'][:]

print("第一层的连接权重矩阵：\n%s\n"%dense_1_kernel)
print("第一层的连接偏重矩阵：\n%s\n"%dense_1_bias)
print("第二层的连接权重矩阵：\n%s\n"%dense_2_kernel)
print("第二层的连接偏重矩阵：\n%s\n"%dense_2_bias)

