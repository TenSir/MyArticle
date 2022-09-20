# -*- coding: utf-8 -*-
import subprocess
import time
import pyautogui
import uiautomation as auto

import open_url_yb
import myLogin
import book_course_task


def woolscrapy(usename,password):
    open_res = open_url_yb.open_url_yb()
    login_res = myLogin.login(usename, password)
    book_click_res = book_course_task.bookTask()
    course_click_res = book_course_task.courseTask()


woolscrapy('166590','lxed1')
woolscrapy('180054','Th152')
woolscrapy('186504','15228@QQ.com')







