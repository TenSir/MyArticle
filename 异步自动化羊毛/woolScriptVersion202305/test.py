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
#
#     import uiautomation as auto


import uiautomation as auto
googleWin = auto.PaneControl(searchDepth=1, Name='异步社区-致力于优质IT知识的出版和分享 - Google Chrome')
print('googleWin:', googleWin)
print('googleWin:', googleWin)
if googleWin.Exists(3, 1):
    googleWin.SetActive(0.5)
    googleWin.Maximize(0.5)
PublicTimeControl = auto.TextControl(Name="出版时间", LocalizedControlType="文本")
PublicTimeControl.Click(waitTime=0.5)




# How found:	Selected from tree...
# Name:	"出版时间"
# ControlType:	UIA_TextControlTypeId (0xC364)
# LocalizedControlType:	"文本"
# BoundingRectangle:	{l:1589 t:402 r:1650 b:420}
# IsEnabled:	true
# IsOffscreen:	false
# IsKeyboardFocusable:	false
# HasKeyboardFocus:	false
# AccessKey:	""
# ProcessId:	29556
# RuntimeId:	[2A.6058A.4.FFFFFE39]
# FrameworkId:	"Chrome"
# IsControlElement:	true
# ProviderDescription:	"[pid:29556,providerId:0x0 Main(parent link):Microsoft: MSAA Proxy (IAccessible2) (unmanaged:UIAutomationCore.DLL)]"
# AriaProperties:	""
# IsPassword:	false
# IsRequiredForForm:	false
# IsDataValidForForm:	true
# HelpText:	""
# Culture:	0
# LegacyIAccessible.ChildId:	0
# LegacyIAccessible.DefaultAction:	"点击祖先实体"
# LegacyIAccessible.Description:	""
# LegacyIAccessible.Help:	""
# LegacyIAccessible.KeyboardShortcut:	""
# LegacyIAccessible.Name:	"出版时间"