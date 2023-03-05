
# class Car():
#     def __init__(self):
#         self.name = 'BMW'
#     def __call__(self,name):
#         lenName = len(name)
#         return lenName


# def fun_1(x):
#     y = 100
#     def inner_fun():
#         print('求商:',x/y)
#     return inner_fun
#
# return_1 = fun_1(200)
# return_1()



# def fun_1(x):
#     def fun_2(y):
#         return x/y
#     return fun_2
#
# result = fun_1(100)
# print(result(200))
# print(result(300))
#
# print(fun_1.__closure__)
# print(result.__closure__)
# print(type(result.__closure__))
# print(result.__closure__[0].cell_contents)
# print(result.__code__.co_freevars)
# print(result.__code__.co_varnames)


# def Test():
#     var = 10
#     def my_add(x):
#        var += 1
#        return var + x
#     return my_add
#
# my_add = Test()
# res = my_add(100)
# print(res)



# def Test():
#     var = 10
#     def my_add(x):
#         nonlocal  var
#         var += 1
#         return var + x
#     return my_add
#
# my_add = Test()
# res = my_add(100)
# print(res)


# var = 10
# def Test():
#     def my_add(x):
#         global var
#         var += 1
#         return var + x
#     return my_add
#
# my_add = Test()
# res = my_add(100)
# print(res)



def func_out(n):
    i = 1
    def func_in():
        nonlocal i
        i = i + n
        print(i)
    return func_in

loop = 0
start = func_out(3)
while loop<3:
    start()
    loop += 1

