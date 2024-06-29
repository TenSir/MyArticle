import time



def mylogger(fun=None,flag='1'):
    def use_time(inner_func):
        def wrapper(*args,**kwargs):
            start_t = time.time()
            myf = inner_func(*args,**kwargs)
            cost_time = time.time()-start_t
            print(f"函数 {inner_func.__name__} 运行的时间为 {cost_time}")
            return myf
        return wrapper
    if fun is None:
        return use_time
    else:
        return use_time(fun)

# 1.第一种情况
@mylogger
def add(a,b):
    c1 = a + b
    print('我是add的操作')
add(1,2)

# 2.第二种情况
@mylogger(flag='Yes')
def add(a,b):
    c1 = a + b
    print('我是add的操作')
add(1,2)

# 3.第三种情况
@mylogger()
def add(a,b):
    c1 = a + b
    print('我是add的操作')
add(1,2)


# _use_time = mylogger(flag = "Yes")
# add_1 = _use_time(add)
# add_1(1,2)
