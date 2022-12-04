
'''
####################  TypeError
def add_number(parameter1, parameter2):
    sum = parameter1 + parameter2
    return sum

total = add_number(1, 2, 3)
print(total)


####################### AttributeError 
class Car():
    def __init__(self):
        # 屏幕设置
        self.car_lenth = 3.5
        self.car_height = 1.7
        self.car_name = 'Benz'
new_car = Car
print(new_car.car_width)


###################### IndexError
def len_list(number):
    print(number[len(number)] + 1)
len_list([1,2,3,4,5])



#################### KeyError
def visit_dic(dic_num):
    print(dic_num['4'])
dic_123 = {'1':'我', '2':'爱', '3':'python学堂'}
visit_dic(dic_123)


#################### NameError

def name_err():
    print(name)
name_err()



#################### OSError

def open_file(filename):
    file=open(filename)
file = r'C:\\Users\TEN\Desktop\'1.txt'
open_file(file)



################  SyntaxError
def sy_error():
    print('python知识学堂')
    print('今天天气好)

sy_error()



########################  多except模式
# ZeroDivisionError
try:
    file = open(r'C:\\Users\TEN\Desktop\'1.txt')  # 桌面不存在文件1.txt
    print(100/0)
    print(100/2)
except ZeroDivisionError as zeror:
    print('分母不能为0')
except FileNotFoundError as fn:
    print(fn)
    print('python')
else:
    print('运行结束')

##########################################################
def trying():
    try:
        try:
            print(100/0)
        except:
            print('分母不能为0')
        try:
            file = open(r'C:\Users\TEN\Desktop\'1.txt')
        except FileNotFoundError as fn:
            print(fn)
    except:
        print('python')
    finally:
        print('最后的打印')
trying()


#########################################################

'''


