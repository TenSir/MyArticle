# Capture full size screenshot
import os
import time
import pyautogui
import pyperclip
from selenium import webdriver
from selenium.webdriver.chrome.options import Options



def get_screen_image(url):
    t1 = time.time()
    driver = webdriver.Chrome()
    driver.maximize_window()
    time.sleep(0.5)
    #控制浏览器写入并转到链接
    driver.get(url)
    time.sleep(0.5)
    # 截图并关掉浏览器
    pyautogui.press('f12')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'shift', 'P')
    time.sleep(0.5)
    pyperclip.copy('Capture full size screenshot')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl','v')
    time.sleep(0.5)
    pyautogui.press('Enter')

    # 等待文件下载完成
    path = r'C:\Users\LEGION\Downloads'
    FLAG = 1
    while FLAG:
        time.sleep(1)
        print('开始查找文件存在不存在')
        for filename in os.listdir(path):
            file = os.path.abspath(path) + '\\' + filename
            if '.png' in file or '.PNG' in file:
                print('文件存在')
                FLAG = 0

    time.sleep(1)
    driver.quit()
    t2 = time.time()
    print(t2-t1)

#你输入的参数
url='https://blog.csdn.net/th1522856954/article/details/109589959?spm=1001.2014.3001.5501'
get_screen_image(url)

