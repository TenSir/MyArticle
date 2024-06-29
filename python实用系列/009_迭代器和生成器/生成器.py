# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 生成器.py
# @Time    : 2020/8/8 19:02
# @Cnblogs ：Python知识学堂

"""
def myGenerator() :
    mylist = range(1,3)
    for i in mylist :
        yield i*i

test_generator = myGenerator()
print('test_generator:',test_generator)
for i in test_generator:
     print(i)

"""



def Fib(n):
    num_1, num_2 = 0, 1
    while num_1 < n:
        yield num_1
        num_1, num_2 = num_2, num_1 + num_2

for n in Fib(4):
    print(n)