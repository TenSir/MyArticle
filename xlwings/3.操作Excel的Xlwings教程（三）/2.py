import xlwings as xw

wb = xw.Book('3.xlsx')
# range=xw.Range ('A3').insert(shift='right')  # 插入单元格

# range = xw.Range('B2:B3').api.merge() # 从指定的Range对象创建一个合并的单元格。
# xw.Range("A4:C4").api.merge()  # 通过pywin32的api调用merge
# range = xw.Range('A1').name  # 设置或获取范围的名称

# range = xw.Range('A1').number_format # 获取并设置Range的number_format

# xw.Range('A1:C3').number_format = '0.00%'
# xw.Range('A1:A3').resize(row_size = None,column_size = None )  # 调整指定范围的大小

# xw.Range ('B2:C4').row_height       # 获取范围的高度（以磅为单位
# xw.Range ('B2:C4').row_height = 15  # 设置范围的高度（以磅为单位
# 以指定的格式返回范围的地址


# range = xw.Range ('B2:C4').rows  # 返回一个RangeRows对象，该对象表示指定范围内的行。


# xw.Range('B2:C4').offset(row_offset=0,column_offset=0) # 选定单元格进行移动
# row_offset行偏移，column_offset列偏移

# range = xw.Range('B2:C4').shape  # 以数组的形式返回所选范围的值

# sheet.api.row('2:4').insert # 插入行，在第2-4行插入空白行

# sheet.api.row('2:4').delete # 删除行
# range = xw.Range('B2:C4').size  # 返回所选范围单元格个数(元素个数)


# range = xw.Range('B2:C4').options(ndim=2).value
range = xw.Range('A1:C1').rows.count
range = xw.Range('A1:C1').columns.count
print(range)
