
'''

def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1
fab(10)



def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        # print(b)
        a, b = b, a + b
        n = n + 1

for n in fab(10):
    print(n)



yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，
Python 解释器会将其视为一个 generator，调用 fab(5) 不会执行 fab 函数，而是返回一个 iterable
对象在 for 循环执行时，每次循环都会执行 fib 函数内部的代码，执行到 yield b 时，fib 函数就返回一
个迭代值，下次迭代时，代码从 yield b 的下一条语句继续执行，而函数的本地变量看起来和上次中断执行前
是完全一样的，于是函数继续执行，直到再次遇到 yield。



string_index = [1,2,3,4,5,6]
for number_1 in range(len(string_index) , 1 ,-1):
    print(number_1)
print('___________________')
for number_2 in range(2, len(string_index) + 1):
    print(number_2)


'''

