import xlwings as xw

"""
app=xw.App(visible=False,add_book=False)
wb=app.books.add()
wb.save('2.xlsx')
wb.close()
app.quit()

"""
# app = xw.apps.active
# app=xw.App(visible=True,add_book=False)
# app.display_alerts=False
# app.screen_updating=False
# wb = xw.books.active
# wb = app.books.open('1.xlsx')
# wb2 = app.books.open('2.xlsx')

app = xw.App(visible=False, add_book=False)
wb = app.books.open('3.xlsx')
app.display_alerts = False

ws = wb.sheets['Sheet1']
shape = ws.used_range.shape
print(shape)

nrow1 = ws.api.UsedRange.Rows.count
ncol1 = ws.api.UsedRange.Columns.count
print(nrow1)
print(ncol1)

rng = ws.range('A1').expand()
nrow2 = rng.last_cell.row
ncol2 = rng.last_cell.column
print(nrow2)
print(ncol2)

print(ws.range('AB2').column)

print(ws.range('AB2').row)

print(ws.range('A1').row_height)
print(ws.range('A1').column_width)

# 获取颜色
print(ws.range('AB2').color)
# 设置颜色,可根据RGB颜色表寻找自己想要的颜色
ws.range('AB2').color = (255, 0, 0)

wb.save()
wb.close()
app.quit()
