

"""
def open_file(filename):
    try:
        with open(filename) as f:
            file = open(filename)
            print('文件名:',file.name)
    except OSError as E:
        print('open file error')

filename = r'C:\\Users\LEGION\Desktop\1.txt'
open_file(filename)



# 自定义一个异常类
class Self_def_exception(Exception):
    def __init__(self,num):
        self.num=num

def Judge(num):
    try:
        num/0
            #print('num能被2整除')
    except Self_def_exception as se:
        print('num不能被2整除')
        #print('se:',se)
    else:
        print("未发生异常")

Judge(3)
#Judge(2)

"""


# 自定义异常类,继承于Exception基类
class Division_Error(Exception):
    # 当输出有误时，抛出此异常
    # 初始化
    def __init__(self, value, name):
        self.value = value
        self.name = name

    # 返回异常类对象的相关信息
    def __str__(self):
        if self.value % 2 != 0 and len(self.name) < 5:
            return ("{}不是偶数，{}长度小于4".format(self.value,self.name) )
try:
    # 抛出异常
    print('尝试显示抛出异常')
    diverr = Division_Error(1,'name')
    raise diverr
    #或raise Division_Error(1,'name')'
except Division_Error as diverr:
    print('diverr: {}'.format(diverr))
