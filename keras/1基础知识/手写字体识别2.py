# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 2.py
# @Time    : 2020/8/15 16:10
# @Cnblogs ：Python知识学堂


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
path = r'F:\kerasdataset\mnist.npz'
(X_train, y_train), (X_test, y_test) = mnist.load_data(path)

X_train = X_train.reshape(len(X_train),-1)
X_test = X_test.reshape(len(X_test), -1)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(512, input_shape=(28*28,),activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.05)
loss, accuracy = model.evaluate(X_test, y_test)
Testloss, Testaccuracy = model.evaluate(X_test, y_test)
print('Testloss:', Testloss)
print('Testaccuracy:', Testaccuracy)

save_path = r'F:\kerasdataset\mnist_test.h5'
model.save(save_path)

"""

model_save_path =r'F:\kerasdataset\mnist_test.h5'
#model.save_weights(model_save_path)
model.load_weights(model_save_path)
# 保存模型网络结构
yaml_save = model.to_yaml()
with open("modelsave.yaml", "w") as f:
	f.write(yaml_save)

from keras.models import model_from_json
from keras.models import model_from_yaml

model_json = model_from_json(json_save)
model_yaml = model_from_yaml(yaml_save)

# 载入模型
model = load_model(r'F:\kerasdataset\mnist_test.h5')
# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=2)

"""