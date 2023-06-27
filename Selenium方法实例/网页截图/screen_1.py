# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def get_screen_image(url, pic_name):
    t1 = time.time()
    # 一定要使用headless就是无界面模式这个模式，不然截不了全页面
    chrome_options = Options()
    chrome_options.add_argument('headless')
    driver = webdriver.Chrome(chrome_options=chrome_options)
    # 打开网页
    driver.get(url)
    time.sleep(1)
    # 通过执行js方法来获取网页的宽度和高度
    width = driver.execute_script("return document.documentElement.scrollWidth") + 10
    height = driver.execute_script("return document.documentElement.scrollHeight") + 10
    print(width,height)
    # 设置无头浏览器宽度和高度
    driver.set_window_size(width, height)
    time.sleep(1)
    #截图并关掉浏览器
    driver.save_screenshot(pic_name)
    driver.quit()
    t2 = time.time()
    print(t2-t1)

#你输入的参数
url='https://blog.csdn.net/th1522856954/article/details/109589959?spm=1001.2014.3001.5501'
pic_name = r'C:\Users\LEGION\Desktop\tweets_code\Selenium方法实例\网页截图\image.png'
get_screen_image(url, pic_name)

#6.889503002166748



#print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
