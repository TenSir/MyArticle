# -*- coding: utf-8 -*-
import subprocess
import time
import uiautomation as auto


def open_url_yb():
    print('root Control:', auto.GetRootControl())
    chromePath = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    url = r'https://www.epubit.com/'
    parameter = '--force-renderer-accessibility'
    startmax = '-start-maximized'

    run_cmd = chromePath + ' ' + url + ' ' + parameter + ' ' + startmax
    subprocess.Popen(run_cmd)

    time.sleep(4)
    # 等待网页加载成功
    loop=1
    mainWindow = auto.DocumentControl(ClassName='Chrome_RenderWidgetHostHWND')
    while "致力于优质IT知识的出版和分享" not in mainWindow.Name:
        print('网页加载成功')
        time.sleep(1)
        loop = loop + 1
        mainWindow = auto.DocumentControl(ClassName='Chrome_RenderWidgetHostHWND')
        if loop == 20:
            print('超时，请检查网络')
            return None
    return 'success'


# s = open_url_yb()
# print('s:',s)
