import time

# def use_time(func):
#     def wrapper(*args,**kwargs):
#         start_t = time.time()
#         myf = func(*args,**kwargs)
#         cost_time = time.time()-start_t
#         print(f"函数 {func.__name__} 运行的时间为 {cost_time}")
#
#     return wrapper
#
# # @ use_time
# def add(a,b):
#     c1 = a + b
#     print('我是add的操作')
# # add(1,2)
#
# wrapper = use_time(add(1,2))
# wrapper(11,1)


##################################################
# def aaa():
#     name = 'egon'
#     def wrapper():
#         money = 1000
#         def tell_info():
#             print('egon have money %s' % money)
#             print('my name is %s' % name)
#         return tell_info
#     return wrapper
#
# wrapper = aaa()
# tell_info = wrapper()
# tell_info()
# print(tell_info.__closure__[0].cell_contents)
# print(tell_info.__closure__[1].cell_contents)
##################################################


################################################# 不含参数的方式
# def mylogger(flag):
#     def use_time(func):
#         def wrapper():
#             start_t = time.time()
#             myf = func()
#             cost_time = time.time()-start_t
#             print(f"函数 {func.__name__} 运行的时间为 {cost_time}")
#             if flag:
#                 print('此次我们会收集函数日志！')
#             return myf
#         return wrapper
#     return use_time
#
# def add():
#     print('我是add的操作')
#
# _use_time = mylogger(flag = "Yes")
# add_1 = _use_time(add)
# add_1()


################################################# 含有参数的方式
# def mylogger(flag):
#     def use_time(func):
#         def wrapper(*args,**kwargs):
#             start_t = time.time()
#             myf = func(*args,**kwargs)
#             cost_time = time.time()-start_t
#             print(f"函数 {func.__name__} 运行的时间为 {cost_time}")
#             if flag:
#                 print('此次我们会收集函数日志！')
#             return myf
#         return wrapper
#     return use_time
#
# def add(a,b):
#     c1 = a + b
#     print('我是add的操作')
#
# _use_time = mylogger(flag = "Yes")
# add_1 = _use_time(add)
# add_1(1,2)

