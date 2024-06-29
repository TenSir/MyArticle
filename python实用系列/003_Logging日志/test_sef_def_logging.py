
import datetime
from sef_def_logger import MyLog

my_logg = MyLog().logger
my_logg.info("代码开始运行的时间{}".format(datetime.datetime.now()))
my_logg.debug('看看debug')
my_logg.error('This is a error')