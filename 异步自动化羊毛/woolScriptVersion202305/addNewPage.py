"""
点击 + 号
打开一个新的页面
"""
# -*- coding: utf-8 -*-
import subprocess
import time
import pyperclip
import pyautogui
import uiautomation as auto

def addnewpage():
    print('root Control:', auto.GetRootControl())
    googleWin = auto.PaneControl(searchDepth=1,Name='异步社区-致力于优质IT知识的出版和分享 - Google Chrome')
    print('googleWin:',googleWin)
    if googleWin.Exists(3,1):
        googleWin.SetActive(0.5)
        googleWin.Maximize(0.5)

    newsheetControl = auto.ButtonControl(Name="新标签页", LocalizedControlType="按钮")
    newsheetControl.Click(waitTime=0.5)

    cursorX, cursorY = auto.GetCursorPos()
    auto.SetCursorPos(cursorX + 15, cursorY + 50)
    pyautogui.click()

    time.sleep(1)
    newUrl = r"https://www.epubit.com/books?code=UB7d76fd4f83658&type=ushu"
    pyautogui.typewrite(newUrl)
    time.sleep(2)
    pyautogui.press("Enter")
    time.sleep(0.5)
    pyautogui.press("Enter")
    time.sleep(0.5)

    return "sucess add newPage"

# addnewpage()