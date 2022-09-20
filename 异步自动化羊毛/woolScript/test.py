# from selenium import webdriver
#
# # 初始化浏览器为chrome浏览器
# browser = webdriver.Chrome()
# # 设置浏览器大小：全屏
# browser.maximize_window()
#
# # 访问百度首页
# browser.get(r'https://www.epubit.com/')
#
# # 关闭浏览器
# browser.close()

# import uiautomation as auto
# googleWin = auto.PaneControl(searchDepth=1, Name='异步社区-致力于优质IT知识的出版和分享 - Google Chrome')
# print('googleWin:', googleWin)
# if googleWin.Exists(3, 1):
#     googleWin.SetActive(0.5)
#     googleWin.Maximize(0.5)
#     # Name: "签到"
#     # ControlType: UIA_GroupControlTypeId(0xC36A)
#     signControl = auto.GroupControl(Name = '签到')
#     signControl.Click(waitTime=0.5)

