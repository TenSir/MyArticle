import subprocess
import time
import pyautogui
import uiautomation as auto

# 每一本书
def click_every_book():
    mainW = auto.PaneControl(SubName='异步社区-致力于优质IT知识的出版和分享')
    print(mainW)
    if mainW.Exists(3,1):
        mainW.SetActive(0.5)
        mainW.Maximize(0.5)

    # 1. 查看点赞元素
    # Name: "Breadcrumb"
    # ControlType: UIA_GroupControlTypeId(0xC36A)
    # LocalizedControlType: "导航"
    GroupCon = auto.GroupControl(Name = 'Breadcrumb',LocalizedControlType = '导航')
    nextControl = GroupCon.GetNextSiblingControl()
    nnextControl = nextControl.GetNextSiblingControl()

    # 点赞和分享元素根元素
    sonControl = nnextControl.GetFirstChildControl()
    FingerShareRootControl = sonControl.GetNextSiblingControl()

    # 双击，间隔2秒钟点击爱心
    FingerRoot = FingerShareRootControl.GetFirstChildControl()
    Fingercontrol = FingerRoot.GetFirstChildControl()
    Fingercontrol.Click(waitTime=1)
    time.sleep(0.6)
    Fingercontrol.Click(waitTime=0.5)

    # 点击分享
    FingerRootNext = FingerRoot.GetNextSiblingControl()
    FingerRootNNext = FingerRootNext.GetNextSiblingControl()
    Sharecontrol = FingerRootNNext.GetFirstChildControl()
    Sharecontrol.Click(waitTime=0.8)

    # 移动鼠标
    cursorX, cursorY = auto.GetCursorPos()
    print(cursorX)
    print(cursorY)
    time.sleep(0.6)
    auto.SetCursorPos(cursorX + 20, cursorY + 50)
    pyautogui.click()

    # 分享界面
    time.sleep(5)
    shareWin = auto.PaneControl(SubName='分享到微博')
    if shareWin:
        shareWin.SendKeys('{Ctrl}w', waitTime=1)
    time.sleep(1)
    shareWin.SendKeys('{Ctrl}w', waitTime=1)


# 每一个课程
def click_every_course():
    mainW = auto.PaneControl(SubName='异步社区-致力于优质IT知识的出版和分享')
    print(mainW)
    if mainW.Exists(3,1):
        mainW.SetActive(0.5)
        mainW.Maximize(0.5)

    # 1. 查看点赞元素
    # Name: "Breadcrumb"
    # ControlType: UIA_GroupControlTypeId(0xC36A)
    # LocalizedControlType: "导航"
    GroupCon = auto.GroupControl(Name = 'Breadcrumb',LocalizedControlType = '导航')
    nextControl = GroupCon.GetNextSiblingControl()
    nnextControl = nextControl.GetNextSiblingControl()

    # 点赞和分享元素根元素
    sonControl = nnextControl.GetFirstChildControl()
    FingerShareRootControl = sonControl.GetNextSiblingControl()

    # 双击，间隔2秒钟点击爱心
    RootControl = FingerShareRootControl.GetChildren()
    # 爱心元素
    Fingercontrol = RootControl[1].GetFirstChildControl()
    Sharecontrol = RootControl[3].GetFirstChildControl()

    # 点击爱心元素
    Fingercontrol.Click(waitTime=1)
    time.sleep(0.6)
    Fingercontrol.Click(waitTime=0.5)

    # 点击分享
    Sharecontrol.Click(waitTime=0.6)

    # 移动鼠标
    cursorX, cursorY = auto.GetCursorPos()
    print(cursorX)
    print(cursorY)
    time.sleep(0.8)
    auto.SetCursorPos(cursorX + 20, cursorY + 50)
    pyautogui.click()

    # 分享界面
    time.sleep(5)
    shareWin = auto.PaneControl(SubName='分享到微博')
    if shareWin:
        shareWin.SendKeys('{Ctrl}w', waitTime=1)
    time.sleep(1)
    shareWin.SendKeys('{Ctrl}w', waitTime=1)


def bookTask():
    # mainW = auto.PaneControl(Name='异步社区-致力于优质IT知识的出版和分享 - Google Chrome')
    mainW = auto.PaneControl(SubName='异步社区-致力于优质IT知识的出版和分享')
    print(mainW)
    if mainW.Exists(3,1):
        mainW.SetActive(0.5)
        mainW.Maximize(0.5)

    # 1.点击"图书"菜单
    bookControl = auto.TextControl(mainW, Name='图书')
    bookControl.Click()
    time.sleep(2)

    # 2.判断“关于我们”元素是否存在
    aboutmeControl = auto.TextControl(mainW,Name='综合排序')
    if aboutmeControl:
        print('关于我们-元素存在网页加载成功')

    # 3.获取元素主元素
    # GroupControlRoot = auto.GroupControl(mainW, foundIndex=3,LocalizedControlType ='导航')
    # print('GroupControlRoot LocalizedControlType:',GroupControlRoot.LocalizedControlType)
    # nextControl = GroupControlRoot.GetNextSiblingControl()
    # print('nextControl LocalizedControlType:',nextControl.LocalizedControlType)
    # nnextControl = nextControl.GetNextSiblingControl()
    # # 图书列表根元素
    # RootBookListControl = nnextControl
    # print(RootBookListControl)

    # 方法2 通过根元素获取
    DocumentCon = auto.DocumentControl(mainW,Name='异步社区-致力于优质IT知识的出版和分享')
    sonControl_1 = DocumentCon.GetFirstChildControl()
    sonControl_2 = sonControl_1.GetFirstChildControl()
    sonControl_3 = sonControl_2.GetFirstChildControl()
    nextControl = sonControl_3.GetNextSiblingControl()
    nnextControl = nextControl.GetNextSiblingControl()
    RootBookListControl = nnextControl.GetNextSiblingControl()

    # 第一个图书名
    # firstBook = RootBookListControl.GetFirstChildControl()

    """获取当前页面的图书数量"""
    BookList = RootBookListControl.GetChildren()
    print('图书数量：',len(BookList))

    loop = 1
    # for eachControl in BookList:
    for i in range(0,12):
        # 点击每一个图书
        BookList[i].GetInvokePattern().Invoke()
        time.sleep(0.6)
        click_every_book()
        loop += 1
        if loop == 11:
            break
    return 'success'


def courseTask():
    # mainW = auto.PaneControl(Name='异步社区-致力于优质IT知识的出版和分享 - Google Chrome')
    mainW = auto.PaneControl(SubName='异步社区-致力于优质IT知识的出版和分享')
    print(mainW)
    if mainW.Exists(3,1):
        mainW.SetActive(0.5)
        mainW.Maximize(0.5)

    # 1.点击"课程"菜单
    bookControl = auto.TextControl(mainW, Name='课程')
    bookControl.Click()
    time.sleep(2)

    # 2.判断“综合排序”元素是否存在
    aboutmeControl = auto.TextControl(mainW, Name='综合排序')
    if aboutmeControl:
        print('关于综合排序-元素存在网页加载成功')
        print('aboutmeControl:',aboutmeControl)

    # 寻找课程元素
    DocumentCon = auto.DocumentControl(mainW,Name='异步社区-致力于优质IT知识的出版和分享')
    # sonControl_1 = DocumentCon.GetFirstChildControl()
    # sonControl_2 = sonControl_1.GetFirstChildControl()
    # sonControl_3 = sonControl_2.GetFirstChildControl()
    # nextControl = sonControl_3.GetNextSiblingControl()
    # n2extControl = nextControl.GetNextSiblingControl()
    # n3extControl = n2extControl.GetNextSiblingControl()
    # firstCourseEle = n3extControl.GetNextSiblingControl()
    # print(firstCourseEle)

    sonControl_1 = DocumentCon.GetFirstChildControl()
    sonControl_2 = sonControl_1.GetFirstChildControl()
    sumSonControl = sonControl_2.GetChildren()
    print('课程元素个数：',len(sumSonControl))
    time.sleep(1)

    # for i in range(3, len(sumSonControl) - 1):
    for i in range(3,12):
        # 点击元素
        sumSonControl[i].GetInvokePattern().Invoke()
        click_every_course()
        time.sleep(0.5)

    # 点击退出按钮
    exitControl = auto.TextControl(mainW,Name='退出')
    exitControl.Click(waitTime=0.3)
    time.sleep(0.6)
    mainW.SendKeys('{Ctrl}w', waitTime=1)
    return 'success'


# book_click_res = bookTask()
# course_click_res = courseTask()
