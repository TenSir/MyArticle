# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 1.py
# @Time    : 2020/8/21 21:37
# @Cnblogs ：Python知识学堂

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
                    Dense(32,  input_shape=(784,)),
                    Activation('relu'),
                    Dense(10),
                    Activation('softmax'),
                ])

"""

model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
"""

# 多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 回归问题
model.compile(optimizer='rmsprop',
              loss='mse')

model.fit(x_train, y_train, batch_size=32, epochs=10,validation_data=(x_val, y_val))