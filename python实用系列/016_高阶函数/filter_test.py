# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: filter_test.py
# @Time    : 2020/7/11 17:06
# @Cnblogs ：Python知识学堂
##########################################
'''
1.
def select_element(string):
    if 'p' in string:
        return True
        # return 1
    else:
        return False
        # return 0
res_1 = filter(select_element,  ['1','2','3','python'])
res_2 = filter(select_element, {'python':1,'知识':2,'学堂':3})
print(list(res_1))
print(list(res_2))

# 输出均为
# ['python']
# ['python']

'''
#######################################

# 自实现filte()函数功能
def select_element(string):
    if 'p' in string:
        return True
        # return 1
    else:
        return False
        # return 0

def filter_test(function,iterable):
    str_1=[]
    for each in iterable:
        if function(each):
            str_1.append(each)
    return str_1

string = ['1','2','3','python']
res1 = filter_test(select_element,string)
res2 = filter_test(lambda x:x=='python',string)
print(res1)
print(res2)