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


woolscrapy('16659003060','lxed152152')
woolscrapy('18005453669','Th1520823@0602')
woolscrapy('18650488276','1522856954@QQ.com')







