# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 1.py
# @Time    : 2020/8/8 17:48
# @Cnblogs ：Python知识学堂

"""
class Fib(object):
    def __init__(self):
        self.num_1, self.num_2 = 0, 1
    def __iter__(self):
        return self
    def __next__(self):
        self.num_1, self.num_2 = self.num_2, self.num_1 + self.num_2
        if self.num_1 > 100:
            raise StopIteration()
        return self.num_1

for each in Fib():
    print (each)




def Fib(n):
    num_1, num_2 = 0, 1
    while num_1 < 100:
        print(num_1)
        num_1, num_2 = num_2, num_1 + num_2
Fib(2)
"""



S = 'python'
S_1 = iter(target)
print(next(S_1)) #p
print(next(S_1)) #y
print(next(S_1)) #t
print(next(S_1)) #h
print(next(S_1)) #o
print(next(S_1)) #h
print(next(S_1)) # Traceback (most recent call last)...StopIteration