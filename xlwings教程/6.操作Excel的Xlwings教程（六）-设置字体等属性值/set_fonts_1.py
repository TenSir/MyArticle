# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: set_fonts_1.py
# @Time    : 2020/10/9 21:02
# @Cnblogs ：Python知识学堂


import xlwings as xw
App = xw.App(visible=False, add_book=False)
wb = App.books.open('test_sets.xlsx')
sheet = wb.sheets('test_sets')

# sheetFont_name = sheet.range('B1').api.Font.Name
# sheetFont_size = sheet.range('B1').api.Font.Size
# sheetFont_bold = sheet.range('B1').api.Font.Bold
# sheetFont_color = sheet.range('B1').api.Font.Color
# sheetFont_FontStyle = sheet.range('B1').api.Font.FontStyle
#
# print('字体名称:', sheetFont_name)
# print('字体大小:', sheetFont_size)
# print('字体是否加粗:', sheetFont_bold)
# print('字体颜色:', sheetFont_color)
# print('字体类型:',sheetFont_FontStyle)
#
# print('-----设置-----')
# sheet.range('B1').api.Font.Name = '微软雅黑'
# sheet.range('B1').api.Font.Size = 20
# sheet.range('B1').api.Font.Bold = True
# sheet.range('B1').api.Font.Color = 0x0000ff         # 设置为红色RGB(255,0,0)
# sheet.range('B1').api.Font.FontStyle = "Bold Italic"
#
#
# sheetFont_name = sheet.range('B1').api.Font.Name
# sheetFont_size = sheet.range('B1').api.Font.Size
# sheetFont_bold = sheet.range('B1').api.Font.Bold
# sheetFont_color = sheet.range('B1').api.Font.Color
# sheetFont_FontStyle =sheet.range('B1').api.Font.FontStyle
#
# print('字体名称:', sheetFont_name)
# print('字体大小:', sheetFont_size)
# print('字体是否加粗:', sheetFont_bold)
# print('字体颜色:', sheetFont_color)
# print('字体类型:',sheetFont_FontStyle)
#

#sheet.range('B1').api.HorizontalAlignment = -4108    # -4108 水平居中
#sheet.range('B1').api.VerticalAlignment = -4130      # -4108 垂直居中
#sheet.range('B1').api.NumberFormat = "0.00"          # 设置单元格的数字格式

"""设置边框"""
# 底部边框，LineStyle = 1直线，设置边框粗细为2
sheet.range('C3').api.Borders(9).LineStyle = 1
sheet.range('C3').api.Borders(9).Weight = 2

# 左边框，LineStyle = 2虚线
sheet.range('C3').api.Borders(7).LineStyle = 2
sheet.range('C3').api.Borders(7).Weight = 2

# 顶边框，LineStyle = 5 双点划线
sheet.range('C3').api.Borders(8).LineStyle = 5
sheet.range('C3').api.Borders(8).Weight = 2

# 右边框，LineStyle = 4 点划线
sheet.range('C3').api.Borders(10).LineStyle = 4
sheet.range('C3').api.Borders(10).Weight = 2

# 从范围中每个单元格的左上角到右下角的边框
sheet.range('C1').api.Borders(5).LineStyle = 1
sheet.range('C1').api.Borders(5).Weight = 2

# 从范围中每个单元格的左下角到右上角的边框
sheet.range('D1').api.Borders(6).LineStyle = 1
sheet.range('D1').api.Borders(6).Weight = 2

#sheet.range('A1:D1').api.merge()
#sheet.range('A1:D1').api.unmerge()

#sheet.range('E1:E8').Formula = "=Rand()"
# 使用公式
#sheet.range('E1').formula='=SUM(C1:D1)'
# 查看公式
#print(sheet.range('E1').formula)

wb.save()
wb.close()
App.quit()



