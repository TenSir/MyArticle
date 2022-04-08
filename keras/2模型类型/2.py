# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 2.py
# @Time    : 2020/8/23 16:26
# @Cnblogs ：Python知识学堂


from keras import Input
from keras import layers
from keras import models

# 构建输入层的张量
input_tensor = Input(shape=(16,))
# 将输入层的返回的input_tensor作为参数输入到第一层网络的Dense中
dense_1 = layers.Dense(32, activation='relu')(input_tensor)
# 将第一层的返回的dense_1作为参数输入到第二层的Dense中
dense_2 = layers.Dense(64, activation='relu')(dense_1)
# 最后一层来构建输出层，并使用10个神经元来进行分类
output_tensor = layers.Dense(10, activation='softmax')(dense_2)
# 将输入层和输出层的张量输入模型中
model = models.Model(input_tensor, output_tensor)

# 显示模型的相关信息
model.summary()