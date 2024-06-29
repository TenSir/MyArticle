import logging

'''
def use_logging(func):
    logger = logging.getLogger('test_namelog')
    def wrapper():
        #logging.warning("%s is running" % func.__name__)
        logging.warning("%s is running" % func.__name__)
        logger.warning("%s is running" % func.__name__)
        return func()  # 把 foo 当做参数传递进来时，执行func()就相当于执行foo()
    return wrapper
def test_logging():
    print('i am foo')

foo = use_logging(test_logging)
# 因为装饰器 use_logging(foo) 返回的是函数对象 wrapper，这条语句相当于
# foo = wrapper
foo()  # 执行foo()就相当于执行 wrapper()

'''


# 我们可以创建嵌套的函数。
# def add_total():
#     print('我爱python知识学堂')
#
# # func_parameter 函数参数
# def test_output(func_parameter):
#     print('输出测试')
#     print(func_parameter())
#
# test_output(add_total)

#result = test_output(add_total)
#print(result)

