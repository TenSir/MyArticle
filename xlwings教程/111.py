
import xlwings as xw

"""
app=xw.App(visible=True,add_book=False)

Apps = xw.apps
print(Apps)

pid = app.pid
print(pid)

count = xw.apps.count
print(count)
"""

app=xw.App(visible=True,add_book=False)
app.display_alerts=False   #不显示Excel消息框
app.screen_updating=False  #关闭屏幕更新,可加快宏的执行速度

count = xw.apps.count
print(count)

wb = app.books.open('1.xlsx')
wb2 = app.books.open('1.xlsx')

count = xw.apps.count
print(count)

# print(wb.fullname)       # 输出打开的excle的绝对路径
wb.save()
wb.close()
app.quit()

