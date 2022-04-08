# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import time

def get_screen_image(url, pic_name):
    t1 = time.time()
    #设置chrome开启的模式，headless就是无界面模式
    #一定要使用这个模式，不然截不了全页面，只能截到你电脑的高度
    chrome_options = Options()
    chrome_options.add_argument('headless')
    driver = webdriver.Chrome(chrome_options=chrome_options)
    #控制浏览器写入并转到链接
    driver.get(url)
    time.sleep(1)
    #接下来是全屏的关键，用js获取页面的宽高，如果有其他需要用js的部分也可以用这个方法
    width = driver.execute_script("return document.documentElement.scrollWidth") + 10
    height = driver.execute_script("return document.documentElement.scrollHeight") + 10
    print(width,height)
    #将浏览器的宽高设置成刚刚获取的宽高
    driver.set_window_size(width, height)
    time.sleep(1)
    #截图并关掉浏览器
    driver.save_screenshot(pic_name)
    driver.close()
    t2 = time.time()
    print(t2-t1)

#你输入的参数
url='https://blog.csdn.net/th1522856954/article/details/109589959?spm=1001.2014.3001.5501'
pic_name = r'C:\Users\LEGION\Desktop\tweets_code\Selenium\网页截图\image.png'
get_screen_image(url, pic_name)

#6.889503002166748



#print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
