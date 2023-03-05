# import functools
#
# def describe(function):
#     @functools.wraps(function)
#     def wrapped(*args, **kwargs):
#         """函数功能描述1"""
#         return function(*args, **kwargs)
#     return wrapped
#
# @describe
# def print_name():
#     """功能描述2"""
#     print('Tom')
#
# print(print_name.__name__)
# print(print_name.__doc__)

#############################################################
# import wrapt
# @wrapt.decorator()
# def describe(function):
#     def wrapper(wrapped, instance, args, kwargs):
#         """函数功能描述1"""
#         return function(*args, **kwargs)
#     return wrapper
#
# @describe
# def print_name():
#     """功能描述2"""
#     print('Tom')
# print(print_name.__name__)
# print(print_name.__doc__)


# import wrapt
# def describe(function):
#     @wrapt.decorator()
#     def wrapper(wrapped, instance, args, kwargs):
#         """函数功能描述1"""
#         return function(*args, **kwargs)
#     return wrapper
#
# @describe
# def print_name(name):
#     """功能描述2"""
#     print(f'my name is {name}')
#
# print(print_name('Tom'))
# print(print_name.__name__)
# print(print_name.__doc__)


# from decorator import decorator
# from datetime import datetime
#
# @decorator
# def logging(func, *args, **kwargs):
#     print(f"[DEBUG] {datetime.now()}: enter {func.__name__}()")
#     return func(*args, **kwargs)
#
# @logging
# def f(a, b):
#     return a + b
#
# f(1,2)
# print(f.__name__)




from decorator import decorator

@decorator
def describe(func,*args, **kwargs):
    print(f"函数名称：{func.__name__}()")
    return func(*args, **kwargs)

@describe
def print_name(name):
    """功能描述2"""
    print(f'my name is {name}')

print_name('Tom')
print(print_name.__name__)
print(print_name.__doc__)
