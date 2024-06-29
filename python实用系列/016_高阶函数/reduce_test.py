# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: reduce_test.py
# @Time    : 2020/7/11 17:44
# @Cnblogs ：Python知识学堂

# 需要导入
'''
#####################################

1.
from functools import reduce

def func(x, y):
    return x + y

string =  [1, 3, 5, 7, 9]
result_1 = reduce(func,string)
result_2 = reduce(func,string,100)
print(result_1)
print(result_2)

res = sum(string)
print(res)


#####################################

def func(x, y):
    return x + y

def self_reduce(function, string, initializer=None):
    it = iter(string)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for each_ele in it:
        value = function(value, each_ele)
    return value

string = {1, 3, 5, 7, 9}
# string = [1, 3, 5, 7, 9]
result_1 = self_reduce(func,string)
result_2 = self_reduce(func,string,100)

print(result_1)
print(result_2)

########################################
'''

def func(x, y):
    return x + y

def self_reduce(function, iterable, initializer=None):
    if initializer is None:
        value =iterable.pop(0)
    else:
        value=initializer
    for each_ele in iterable:
        value=function(value,each_ele)
    return value

string = {1, 3, 5, 7, 9}
result_1 = self_reduce(func,string,100)

print(result_1)


