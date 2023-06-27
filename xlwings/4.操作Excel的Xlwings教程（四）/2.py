'''
app = xw.App(visible=False, add_book=False)
wbk = app.books.open('2.xlsx')
sht = wbk.sheets['sheet1']

title = 'xiaomaomi'
sht.charts['图表 4'].api[1].ChartTitle.Text = title

wbk.save()
wbk.close()
app.quit()


import xlwings as xw
wb = xw.Book('2.xlsx')
sheet = wb.sheets[0]
s = sheet.charts.count   # 输出2
print(s)

'''
# 引用chart有两种方法

# print(sheet.charts[0])  #   法一，输出<Chart 'Chart 1' in <Sheet [2.xlsx]Sheet1>>
# print(sheet.charts['图表 2']) # 法二，输出<Chart 'Chart 2' in <Sheet [2.xlsx]Sheet1>>
# print(sheet.charts[0].api) #返回所使用引擎的本机对象（)


# print(sheet.charts[0].chart_type)   # 返回并设置图表的图表类型,line
# print(sheet.charts[1].chart_type)   # 返回并设置图表的图表类型,bar_clustered


# sheet.charts[0].delete()  # 删除图表。

# sheet.charts[0].height   # 返回或设置代表图表高度
# sheet.charts[0].left    # 返回或设置代表图表水平位置
# sheet.charts[0].top     # 返回或设置代表图表垂直位置
# sheet.charts[0].width  # 返回或设置代表图表宽度


# print(sheet.charts[0].name)  # 返回或设置图表名称，输出Chart 1
# print(sheet.charts[0].parent )# 返回图表的父级对象，输出<Sheet [2.xlsx]Sheet1>


# sheet.charts[0].set_source_data()# 设置图表的源数据范围。


# chart = sheet.charts.add(100,100)  # 会在所选sheet上(100,100)位置新建一个图表(空白表)

# left (float, default 0):left position in points
# top (float, default 0):top position in points
# width (float, default 355):width in points
# height (float, default 211):height in points


'''
import xlwings as xw
sheet = xw.Book().sheets[0]
sheet.range('A1').value = [['变量1', '变量2'], [1,10000]]
chart = sheet.charts.add()
chart.set_source_data(sheet.range('A1').expand())  #
chart.chart_type = 'line'                          # 设置图标的类型

'''

import xlwings as xw

sheet = xw.Book('4.xlsx').sheets[0]
chart = sheet.charts.add()  # 新增chart
chart.set_source_data(sheet.range('A1').expand())  # 数据源：sheet.range('A1:B7')，或者sheet.range('A1').expand()
chart.chart_type = 'line'  # 设置图标的类型，此处为线型，具体的类型查看office官网VBA操作的手册
title = 'python知识学堂粉丝数'  # 标题名称初始化
chart.api[1].SetElement(2)
chart.api[1].ChartTitle.Text = title  # 设置标题名称
chart.api[1].SetElement(302)
chart.api[1].Axes(1).AxisTitle.Text = "日期"  # 横轴标题名称
chart.api[1].SetElement(311)
chart.api[1].Axes(2).AxisTitle.Text = "粉丝数"  # 纵轴标题名称

# Charts([<Chart 'Chart 1' in <Sheet [2.xlsx]Sheet1>>, <Chart 'Chart 2' in <Sheet [2.xlsx]Sheet1>>])
