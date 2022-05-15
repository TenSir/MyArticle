import xlwings as xw
wb = xw.Book('1.xlsx')
sheet = wb.sheets[0]
#wb.sheets[0].shapes  # Shapes([<Shape 'Rectangle 1' in <Sheet [1.xlsx]Sheet1>>])

#sheet = wb.sheets[0]
#shape = sheet.shape
print(wb.sheets[0].shapes.name)

#activate()
#ss = xw.Shape.activate()
#ss =wb.sheets['sheet1'].shapes[0].activate()
#print(ss)




#print(xw.Shape(1).activate)      # 激活里面的一个shape
# 或者 sheet.shapes[0].activate

#print(sheet.shapes[0].height)
#print(xw.Shape(1).parent)    # parent 返回形状的父级,输出<Sheet [1.xlsx]Sheet1>
#xw.Shape(1).left  # left 返回或设置表示形状水平位置的点数。

#print(xw.Shape(1).activate)      # 激活里面的一个shape
# 或者 sheet.shapes[0].activate

#print(xw.Shape(1).delete)

#print(sheet.shapes[0].top) # top 返回或设置表示形状垂直位置的点数
#print(sheet.shapes[0].width) # width 返回或设置表示形状宽度的点数。
#print(sheet.shapes[0].type)  # type 返回形状的类型。输出 auto_shape

