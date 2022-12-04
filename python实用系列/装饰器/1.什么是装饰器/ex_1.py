# import time
#
# def add_total(name,a,b):
#     print('传递的参数为：',name)
#     def add1(a,b):
#         t1 = time.time()
#         c1 = a + b
#         t2 = time.time()
#         print("add1函数，花费的时间是：{}".format(t2-t1))
#         return '我是+1的操作'
#     def add2(a,b):
#         t1 = time.time()
#         c2 = a + b
#         t2 = time.time()
#         print("add2函数，花费的时间是：{}".format(t2 - t1))
#         return '我是+2的操作'
#     def add3(a,b):
#         t1 = time.time()
#         c3 = a + b
#         t2 = time.time()
#         print("add2函数，花费的时间是：{}".format(t2 - t1))
#         return '我是+3的操作'
#
#     add1(a, b)
#     add2(a, b)
#     add3(a, b)
#
# add_total('Hello',1,3)


# def add_total(name):
#     print('传递的参数为：',name)
#     def add1():
#         return '我是+1的操作'
#     def add2():
#         return '我是+2的操作'
#     def add3():
#         return '我是+3的操作'
#     if "Hello" in name:
#         return add3()
#
# res = add_total("Hello Python知识学堂")
# print(res)


# def add_total():
#     print('这是总和的计算')
#
# def test_output(func_parameter):
#     print('输出测试')
#     print(func_parameter())
#     return "success"
#
# res = test_output(add_total)
# print(res)


# def use_time(func):
#     def wrapper(*args,**kwargs):
#         start_t = time.time()
#         myf = func(*args,**kwargs)
#         cost_time = time.time()-start_t
#         logging.warning(f"函数 {func.__name__} 运行所用的时间为 {cost_time}")
#         return myf
#     return wrapper