# from functools import wraps
# from functools import update_wrapper
#
# class Car:
#     def __init__(self, func):
#         #update_wrapper(self,func)
#         self.func = func
#
#     def __call__(self, *args, **kwargs):
#         print('Log，start Driving')
#         callRes = self.func(*args, **kwargs)
#         return callRes
# @Car
# def use(carName, user):
#     word = f'{carName} belongs to {user}'
#     return word
#
#
# res = use('BMW', 'Tom')
# print(res)
# print(type(use))
# print(use.__name__)
################################################################

# from functools import wraps
# from functools import update_wrapper
#
# class Car:
#     def __init__(self, func):
#         update_wrapper(self,func)
#         self.func = func
#
#     def __call__(self, *args, **kwargs):
#         print('start Driving')
#         callRes = self.func(*args, **kwargs)
#         return callRes
#
# @Car
# def use(carName, user):
#     word = f'{carName} belongs to {user}'
#     return word
#
# res = use('BMW', 'Tom')
# print(res)
# print(type(use))
# print(use.__name__)
################################################################


import functools

class Car:
    def __init__(self, func,sleep=1):
        functools.wraps(func)(self)
        self.func = func
        self.sleep = sleep

    def __call__(self, *args, **kwargs):
        print(f'start Driving,wait time is {self.sleep}')
        callRes = self.func(*args, **kwargs)
        return callRes

def sleep_time(**kwargs):
    return functools.partial(Car,**kwargs)

@sleep_time(sleep=2)
def use(carName, user):
    word = f'{carName} belongs to {user}'
    return word


res = use('BMW', 'Tom')
print(res)
print(type(use))
print(use.__name__)
