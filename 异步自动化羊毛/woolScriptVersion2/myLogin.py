import subprocess
import time
import uiautomation as auto


def login(user,passw):
    googleWin = auto.PaneControl(searchDepth=1,Name='异步社区-致力于优质IT知识的出版和分享 - Google Chrome')
    print('googleWin:',googleWin)
    if googleWin.Exists(3,1):
        googleWin.SetActive(0.5)
        googleWin.Maximize(0.5)
        # handle = googleWin.NativeWindowHandle
        # auto.SetWindowTopmost(handle, 'True')
        # auto.SwitchToThisWindow(handle)
        # auto.ShowWindow(handle, auto.SW.Maximize)


    mainWindow = auto.DocumentControl(ClassName='Chrome_RenderWidgetHostHWND')
    print('mainWindow:',mainWindow.Name)

    # 1.获取“登录按钮”并点击
    loginControl = auto.TextControl(mainWindow, Name='登录')
    loginControl.Click()

    # 2.'登录'的页面窗口
    loginWindow = auto.DocumentControl(Name='登录')
    print('loginWindow:',loginWindow.Name)
    if "登录" in loginWindow.Name:
        print("登录页面获取成功")

    # 3.获取“输入账号”的控件
    editUseControl = auto.EditControl(loginWindow, Name='请输入手机号码或邮箱')
    editUseControl.Click()
    editUseControl.SendKeys(user, waitTime=0)

    # 4.获取“密码”的控件
    editUseControl = auto.EditControl(loginWindow, Name='请输入密码')
    editUseControl.Click()
    editUseControl.SendKeys(passw, waitTime=0)

    # 5.获取“登录按钮”
    enterloginControl = auto.ButtonControl(loginWindow, Name='登录')
    enterloginControl.Click()

    # 判断退出按钮是否存在
    mainW = auto.PaneControl(searchDepth=1,Name='异步社区-致力于优质IT知识的出版和分享 - Google Chrome')
    loop=1
    # exitControl = auto.TextControl(mainW, Name='退出')
    # while "退出" not in exitControl.Name:
    #     time.sleep(1)
    #     loop = loop + 1
    #     exitControl = auto.TextControl(mainW, Name='退出')
    #     if loop == 20:
    #         print('超时-登录失败，请检查网络')
    #         return None
    # time.sleep(0.5)

    signControl = auto.GroupControl(Name = '签到')
    signControl.Click(waitTime=0.5)
    time.sleep(1)

    return 'loginSuccess'


# user = '16659003060'
# passw = 'xxxx'
# loginStatue = login(user,passw)
# print(loginStatue)

