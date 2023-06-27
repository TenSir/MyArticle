import xlwings as xw

sheet = xw.Book('1.xlsx').sheets[0]
chart = sheet.charts.add()
# 数据源：sheet.range('A1:C7')，或者sheet.range('A1').expand()
chart.set_source_data(sheet.range('A1').expand())
chart.chart_type = 'line'        # 线形
title='商品销量'                  # 标题名称
chart.api[1].SetElement(2)       # 显示标题
chart.api[1].ChartTitle.Text =title    #设置标题名称
chart.api[1].SetElement(302)           # 在轴下方显示主要类别轴标题。
chart.api[1].Axes(1).AxisTitle.Text = "日期"   #横轴标题名称
chart.api[1].SetElement(311)           # 在轴旁边显示主要类别的轴标题。
chart.api[1].Axes(2).AxisTitle.Text = "销量" #纵轴标题名称
chart.api[1].Legend.Position = -4107

# 设置图标的类型，此处为线型，具体的类型查看office官网VBA操作的手册
# https://docs.microsoft.com/zh-cn/office/vba/api/excel.xllegendposition
