# coding = utf-8
import requests

def retry(func):
     def wrapper(*args, **kwargs):
          trytime = 3
          while trytime > 0:
               try:
                    print("执行的函数名称为：",func.__name__)
                    return func(*args, **kwargs)
               except Exception as e:
                    print("Exception ", repr(e))
                    trytime -= 1
          if trytime <=0:
               print('此次打开出错：error')
     return wrapper


@retry
def test_url(url):
    res = requests.get(url, timeout=1)
    data = res.json()
    return data

url = "http://www.baidudududdu.com"
test_url(url)

