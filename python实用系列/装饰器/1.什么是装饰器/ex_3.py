# import logging
# import time


class truck(object):
    def __init__(self,func):
        self.func = func
        print('这是一个初始化的操作')
    # call函数
    def __call__(self):
        # 装饰函数
        print('回家驾驶卡车')
        print('[INFO]:the function {}() is running...'.format(self.func.__name__))
        self.func()
@truck
def smallcar():
    print(f'Tony 的 mini car')
smallcar()



# class Buy(object):
#     def __init__(self, func):
#         self.func = func
#     # 在类里定义一个函数
#     def clothes(func):  # 这里不能用self,因为接收的是body函数
#         # 其它都和普通的函数装饰器相同
#         def ware(*args, **kwargs):
#             print('This is a decrator!')
#             return func(*args, **kwargs)
#
#         return ware
#
# @Buy.clothes
# def body(hh):
#     print('The body feels could!{}'.format(hh))
# body('hh')



# class Subject(object):
#     def __init__(self,name):
#         self.name=name
#     # def __call__(self):
#     #     print("hello "+ self.name)
#     def teacher(self):
#         print("hello my teacher")
#
# a = Subject('Python')
# # 1.第一种方式调用
# print(a.teacher())
# # 2.第二种方式调用
# print(a())


# import pandas
# print(dir(pandas.read_csv))

# class logger(object):
#     def __init__(self,func):
#         self.func=func
#
#     def __call__(self, *args, **kwargs):
#         print('[INFO]:the function {}() is running...'.format(self.func.__name__))
#
#         return self.func(*args, **kwargs)
#
# @logger
# def say(sth):
#     print('say {}!'.format(sth))
#
# say('hello')