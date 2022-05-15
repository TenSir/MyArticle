# -*- coding: utf-8 -*-
# @Author  : TenSir
# @FileName: 2.py
# @Time    : 2020/10/11 9:21
# @Cnblogs ：Python知识学堂
import xlwings as xw
App = xw.App(visible=False, add_book=False)
wb = App.books.open('1.xlsx')
sheet = wb.sheets('Sheet1')

sheetFont_background = sheet.range('A1').api.Font.Background
print('sheetFont_background:' ,sheetFont_background)

#wb.save()
wb.close()
App.quit()
