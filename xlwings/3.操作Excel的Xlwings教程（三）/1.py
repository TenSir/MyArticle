import xlwings as xw

# wb = xw.Book('3.xlsx',read_only=True)
app = xw.App(visible=False, add_book=False)
wb = app.books.open('3.xlsx')
sheet = wb.sheets[0]

sheet.range('A1')  # 引用A1单元格
sheet.range('A1').value  # 取A1单元格的值

var = sheet.range('A1:B2').value  # 引用区域并取值，输出[[1.0, 9.0], [2.0, 10.0]]，以二元列表形式展开
var1 = sheet.range(('A1'), ('B2')).value
print('111')
print(var1)

# ss = sheet.range('A9').add_hyperlink(address='www.baidu.com', "百度") # 添加超链接
# ss = sheet.range('A10').address  # 返回表示范围参考的字符串值，输出 $A$10
# ss = sheet.range('A1').api # 返回所使用引擎的本机对象


# ss = sheet.range('A1').autofit() # 自动调整范围内所有单元格的宽度和高度。

# 要仅自动调整列的宽度，请使用 sheet.range('A1:B2').columns.autofit()
# 要仅自动调整行的高度，请使用 sheet.range('A1:B2').rows.autofit()
# print(ss)

# sheet.range('A1').clear()  # 清除范围的内容和格式
# sheet.range('A1').clear_contents()   # 清除范围的内容，但保留格式。


# ss = sheet.range('A1').color #获取并设置指定范围的背景色。
# sheet.range('A1').color = (255,255,255)
# sheet.range('A1').color = None


# ss = sheet.range('B1:C4').column
# ss = sheet.range('A1:B2').count
# ss = sheet.range('A1').current_region
# ss = sheet.range('A1').delete() # 删除单元格A1,有参数left和up,delete('up')如果省略，Excel将根据范围的形状进行决定。
# ss = sheet.range('A1').end('down') # 返回一个Range对象，该对象表示包含源范围的区域末尾的单元格<Range [3.xlsx]Sheet1!$A$8>


# 获取公式或者输入公式
# sheet.range('A9').formula='=SUM(B1:B5)' # 设置A9单元格公式计算的值
# print(sheet.range('A9').formula)       # 输出'=SUM(B1:B5)'

ss = sheet.range('A1').get_address()

# print(ss)


wb.save()
app.quit()

"""

def copy(self, sheet_name, range_col_row): 
# 复制 sheet_name
是指复制的sheet名字 range_col_row 复制的范围
#复制单元格 sheet.copy(‘sheet1’,‘A1’)
#复制范围 sheet.copy(‘sheet1’,‘A1：D3’)

"""
