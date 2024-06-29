import logging
import time

def use_time(func):
    def wrapper(*args,**kwargs):
        start_t = time.time()
        myf = func(*args,**kwargs)
        cost_time = time.time()-start_t
        print(f"函数 {func.__name__} 运行所用的时间为 {cost_time}")
        return myf
    return wrapper

def add_total(name,a,b):
    print('传递的参数为：',name)
    @use_time
    def add1(a,b):
        c1 = a + b
        return '我是add1的操作'

    @use_time
    def add2(a,b):
        c2 = a + b
        return '我是add2的操作'

    @use_time
    def add3(a, b):
        c3 = a + b
        return '我是add3的操作'
    add1(a, b)
    add2(a, b)
    add3(a, b)

add_total('Hello',1,3)