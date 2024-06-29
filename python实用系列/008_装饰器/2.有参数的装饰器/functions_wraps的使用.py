# import time
# def use_time(func):
#     def wrapper(*args,**kwargs):
#         start_t = time.time()
#         myf = func(*args,**kwargs)
#         cost_time = time.time()-start_t
#         print(f"函数 {func.__name__} 运行的时间为 {cost_time}")
#         return myf
#     return wrapper
#
# @use_time
# def add(a,b):
#     c1 = a + b
#     print('我是add的操作')
#
# add(1,2)
# print(add.__name__)
# print(add.__doc__)



##########################################
# 2.使用 @wrap来进行装饰
import time
from functools import wraps

def use_time(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start_t = time.time()
        myf = func(*args,**kwargs)
        cost_time = time.time()-start_t
        print(f"函数 {func.__name__} 运行的时间为 {cost_time}")
        return myf
    return wrapper

@use_time
def add(a,b):
    """
    用来完成加法操作
    """
    c1 = a + b
    print('我是add的操作')

add(1,2)
print(add.__name__)
print(add.__doc__)

