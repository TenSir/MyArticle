

# __________________________________________________________demo1
class People:
    # 类构造方法，属于实例方法
    def __init__(self):
        self.name = "Python知识学堂"
        self.writeage = 5

    # 静态方法
    @staticmethod
    def myname(name):
        print("my name is",name)

# a = People()
# a.myname('Python知识学堂')
# People.myname('Python知识学堂')


# __________________________________________________________demo2


# class People:
#     # 类构造方法，属于实例方法
#     def __init__(self):
#         self.name = "Python知识学堂"
#         self.writeage = 5
#
#     # 静态方法
#     @staticmethod
#     def myname(name):
#         print("my name is",name)
#
#     # 类的实例方法
#     def language(self):
#         print('我的写作年龄是',self.writeage)
#
#     def languagetwo():
#         print('我的第二语言是英语')

# a = People()
# a.language()

# a = People()
# People.language(a)

###################################
# a = People()
# a.languagetwo()

# People.languagetwo()



# __________________________________________________________demo3
# class People:
#     # 类构造方法，属于实例方法
#     def __init__(self):
#         self.name = "Python知识学堂"
#         self.writeage = 5
#
#     myaddress = '福建省厦门市思明区'
#     # 静态方法
#     @staticmethod
#     def myname(name):
#         print("my name is",name)
#
#     # 类的实例方法
#     def language(self):
#         print('我的写作年龄是',self.writeage)
#
#     # 类方法
#     @classmethod
#     def address(cls):
#         print('正在执行类方法为:', cls)
#         print('正在执行类方法，输出的地址为:',cls.myaddress)


# 使用类名直接调用类方法
# People.address()

# 使用类对象调用类方法
# a = People()
# a.address()



# __________________________________________________________demo4
# class People():
#     # 初始化类参数
#     sum = 0
#
#     # 构造函数
#     def __init__(self):
#         # 自增统计,使用类进行访问
#         People.sum += 1
#
#     @classmethod
#     def how_many(cls):
#         print("创建的对象总数为：", cls.sum)
#
# # 创建工具对象
# a = People()
# b = People()
# c = People()
#
# # 调用类方法
# c.how_many()
# People.how_many()
