
import logging
import os
from logging import config
import yaml


def use_yaml_config(default_path='.\mylog\config.yaml', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding = 'utf-8') as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

if '__main__' == __name__:

    use_yaml_config(default_path='.\mylog\logtest.yaml')
    # root的logger
    root = logging.getLogger()
    # 子记录器的名字与配置文件中loggers字段内的保持一致
    # loggers:
    #   my_testyaml:
    #       level: DEBUG
    #       handlers: [console, info_file_handler,error_file_handler]
    my_testyaml = logging.getLogger("my_testyaml")
    print("rootlogger:", root.handlers)
    print("selflogger", my_testyaml.handlers)
    # 判断子记录器与根记录器的handler是否相同
    print(root.handlers[0] == my_testyaml.handlers[0])

    my_testyaml.info("INFO")
    my_testyaml.C
    my_testyaml.debug("rootDEBUG")

    root.info("INFO")
    root.error('ERROR')
    root.debug("rootDEBUG")




'''

def log_config():
    # '读取日志配置文件'
    path = r'C:\\Users\LEGION\Desktop\推文写作\python实用系列\Logging日志\log_config.conf'
    if os.path.exists(path):
        with open(path,"r",encoding = 'utf-8') as f:
            logging.config.fileConfig(f)
    # 创建一个日志器logger
    logger = logging.getLogger(name="fileLogger")
    rotating_logger = logging.getLogger(name="rotatingFileLogger")

    logger.debug('debug')
    logger.info('info')
    logger.warning('warn')
    logger.error('error')
    logger.critical('critical')

    rotating_logger.debug('debug')
    rotating_logger.info('info')
    rotating_logger.warning('warn')
    rotating_logger.error('error')
    rotating_logger.critical('critical')


#log_config()


'''

'''

    with open(file="./logtest.yaml", mode='r', encoding="utf-8")as file:
        logging_yaml = yaml.load(stream=file, Loader=yaml.FullLoader)
        # print(logging_yaml)
        # 配置logging日志：主要从文件中读取handler的配置、formatter（格式化日志样式）、logger记录器的配置
        logging.config.dictConfig(config=logging_yaml)
    # 获取根记录器：配置信息从yaml文件中获取
    root = logging.getLogger()
    # 子记录器的名字与配置文件中loggers字段内的保持一致
    my_testyaml = logging.getLogger("my_testyaml")
    print("rootlogger:", root.handlers)
    print("selflogger", my_testyaml.handlers)
    # print("子记录器与根记录器的handler是否相同：", root.handlers[0] == my_module.handlers[0])
    my_testyaml.error("DUBUG")
    root.info("INFO")
    root.error('ERROR')
    root.debug("rootDEBUG")
'''