# from openpyxl import Workbook
# # 创建一个工作簿对象
# wb = Workbook()
# # 在索引为0的位置创建一个名为mytest的sheet页
# ws = wb.create_sheet('mytest',0)
# # 对sheet页设置一个颜色（16位的RGB颜色）
# ws.sheet_properties.tabColor = 'ff72BA'
# # 将创建的工作簿保存为Mytest.xlsx
# wb.save('Mytest.xlsx')
# # 最后关闭文件
# wb.close()



from openpyxl import load_workbook
# 加载工作簿
wb2 = load_workbook('Mytest.xlsx')
# 获取sheet页
ws2 = wb2['mytest']

cell_1 = ws2['A1']
cell_2 = ws2.cell(row=1, column=1)

value_1 = ws2['A1'].value
value_2 = ws2.cell(row=1, column=1).value

# 访问A1至C3范围单元格
cell_range = ws2['A1':'C3']

# 访问A列所有存在数据的单元格
colC = ws2['A']
# 访问A列到C列所有存在数据的单元格
col_range = ws2['A:C']
# 访问第1行所有存在数据的单元格
row10 = ws2[1]
# 访问第1行至第5行所有存在数据的单元格
row_range = ws2[1:5]

print(type(cell_range))
print(type(colC))
print(type(col_range))
print(type(row10))
print(row_range)

for each_cell in cell_range:
    for each in each_cell:
     print(each.value)

for each_cell in colC:
    print(each_cell.value)



# for row in ws2.iter_rows(min_row=1, max_col=2, max_row=2):
#     for cell in row:
#         print(cell)
#
# print('111111111111111111111111111111')
# for col in ws2.iter_cols(min_row=1, max_col=2, max_row=2):
#          for cell in col:
#              print(cell)

print(ws2.rows)
print(tuple(ws2.rows))
print(tuple(ws2.columns))
wb2.close()
