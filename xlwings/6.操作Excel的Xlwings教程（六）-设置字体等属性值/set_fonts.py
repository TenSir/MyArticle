# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: set_fonts.py
# @Time    : 2020/10/8 9:31
# @Cnblogs ：Python知识学堂


import xlwings as xw
App = xw.App(visible=False, add_book=False)
wb = App.books.add()
sheet = wb.sheets.add('test_sets')
# Expands the range according to the mode provided. Ignores empty top-left cells (unlike ``Range.end()``).
# 将第一行置为1,2,3,4
# 将第2行开始的第一列置为11,12,13,14
sheet.range(1, 1).expand('right').value = [1, 2, 3, 4]
sheet.range(2, 1).expand('down').options(transpose=True).value = [10, 20, 30, 40]

wb.save('test_sets.xlsx')

wb.close()
App.quit()
