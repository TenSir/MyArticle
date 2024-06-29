class LogUtil:
    """
    封闭简单的Log通用方法，同时将LOG输出到控制台和文件
    """
    def __init__(self, logger_name, log_level, log_file="log.txt"):
        """
        :param logger_name: Logger日志器的名称
        :param log_level: 日志水平
        :param log_file: 日志保存的文件（包括路径+文件）
        """
        self.logger_name = logger_name
        self.log_file = log_file
        self.log_level = log_level

    def operate(self):
        # 创建logger对象
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)

        # 创建处理器，输出日志到log_file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)

        # 创建处理器，输出日志到控制台
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.log_level)

        # 定义处理器的输出格式
        fmt_log = '%(asctime)s %(name)s %(levelname)s %(message)s'
        date_fmt = '%Y%m%d %H:%M:%S'
        formatter = logging.Formatter(fmt=fmt_log, datefmt=date_fmt)
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # 添加处理器到logger对象
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

# 实例调用log方法：计算5除以0，若出现异常时捕获异常
def div_calcul():
    logger = LogUtil(logger_name="divisionCal", log_level=logging.DEBUG, log_file="D://my.log").operate()
    try:
        result = 5 / 0
        print(result)
    except Exception as e:
        logger.critical("[calculation error：%s]" % e)
div_calcul()
"""结果
20190211 17:03:58 divisionCal CRITICAL [calculation error：division by zero]
"""
