# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 1.py
# @Time    : 2020/7/10 21:28
# @Cnblogs ：Python知识学堂

'''
################################################
1.
#参数为函数

def print_1():
    print("我爱python")
def print_2(print_1):
    # 调用print_1()函数
    print_1()
    print("我爱python知识学堂")

print_2(print_1)

##############################################



#############################################
2.
#返回值为函数
def print_1():
    print("我爱python")
def print_2():
    print("我爱python知识学堂")
    # 返回值为一个函数
    return(print_1())

print_2()

###########################################

string_1 = 'Python知识学堂'
string_2 = [1,2,3,4,5,6]
string_3 = {'python':2,'学习':3,1:4}
res1 = map(str,string_1)
res2 = map(str,string_2)
res3 = map(str,string_3)

print(string_1)
print(list(res1))
print(list(res2))
print(list(res3))

###########################################

def func(x,y,z):
    return x**2, y**2, z**2

List1 = [1, 2]
List2 = [1, 2, 3, 4]
List3 = [1, 2, 3, 4, 5]
res = map(func, List1, List2, List3)
print(list(res))

#输出 [(1, 1, 1), (4, 4, 4)]

############################################
def add_test(x):
    return x + 2

string = [1,2,3,4,5,6]
res = map(add_test, string)
print(list(res))

# 结果如下:
[1, 4, 9, 16, 25]

#################################
#for循环来取出内容
def add_test(x):
    return x + 2

string = [1,2,3,4,5,6]
res = map(add_test, string)

res_ls=[]
for i in res:
    res_ls.append(i)
print(res_ls)

#输出为[1, 4, 9]
##################################

'''
def add_test(x):
    return x + 2
# 实现map()函数功能
def self_map(function,iterable):
    str_1=[]
    for each in iterable:
        each_num = function(each)
        str_1.append(each_num)
    # 将str_1转为迭代器对象
    return str_1.__iter__()

string = [1,2,3,4,5,6]
result = self_map(add_test,string)
print(list(result))


#输出为[1, 4, 9]