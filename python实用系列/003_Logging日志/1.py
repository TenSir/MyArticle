
################################################################
'''
import logging
def log_testing():
    selflogger = logging.getLogger('THIS-LOGGING')
    logging.basicConfig(filename='log.txt',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.ERROR)
    selflogger.warning('waring，用来用来打印警告信息')
    selflogger.error('error，一般用来打印一些错误信息')
    selflogger.critical('critical，用来打印一些致命的错误信息，等级最高')

log_testing()
'''

################################################################


################################################################

# 文件和控制台设置

import logging

def log_file():
    log_file = 'testfun.log'
    handler_test = logging.FileHandler(log_file) # stdout to file
    handler_control = logging.StreamHandler()    # stdout to console
    handler_test.setLevel('ERROR')               # 设置ERROR级别
    handler_control.setLevel('INFO')             # 设置INFO级别

    selfdef_fmt = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(selfdef_fmt)
    handler_test.setFormatter(formatter)
    handler_control.setFormatter(formatter)

    logger = logging.getLogger('updateSecurity')
    logger.setLevel('DEBUG')     #设置了这个才会把debug以上的输出到控制台

    logger.addHandler(handler_test)    #添加handler
    logger.addHandler(handler_control)
    logger.info('info,一般的信息输出')
    logger.warning('waring，用来用来打印警告信息')
    logger.error('error，一般用来打印一些错误信息')
    logger.critical('critical，用来打印一些致命的错误信息，等级最高')

log_file()



################################################################
# https://www.jb51.net/article/88449.htm


################################################################
'''
import logging
from logging.handlers import RotatingFileHandler

def logging_fun():
    # 创建日志的记录等级设
    logging.basicConfig(level=logging.DEBUG)
    # 创建日志记录器，指明日志保存的路径，每个日志文件的最大值，保存的日志文件个数上限
    log_handle = RotatingFileHandler("log.txt", maxBytes=1024 * 1024 * 100, backupCount=10)
    # 创建日志记录的格式
    formatter = logging.Formatter("format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',")
    # 为创建的日志记录器设置日志记录格式
    log_handle.setFormatter(formatter)
    # 为全局的日志工具对象添加日志记录器
    logging.getLogger().addHandler(log_handle)
    logging.warning('waring，用来用来打印警告信息')
    logging.error('error，一般用来打印一些错误信息')
    logging.critical('critical，用来打印一些致命的错误信息，等级最高')


    logging.config.fileConfig()
logging_fun()
'''

[loggers]
keys=root,fileLogger,rotatingFileLogger

[handlers]
keys=consoleHandler,fileHandler,rotatingFileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_fileLogger]
level=INFO
handlers=fileHandler
qualname=fileLogger
propagate=0

[logger_rotatingFileLogger]
level=INFO
handlers=consoleHandler,rotatingFileHandler
qualname=rotatingFileLogger
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=INFO
formatter=simpleFormatter
args=("logs/fileHandler_test.log", "a")

[handler_rotatingFileHandler]
class=handlers.RotatingFileHandler
level=WARNING
formatter=simpleFormatter
args=("logs/rotatingFileHandler.log", "a", 10*1024*1024, 5)

[formatter_simpleFormatter]
format=%(asctime)s - %(module)s - %(levelname)s -%(thread)d : %(message)s
datefmt=%Y-%m-%d %H:%M:%S
