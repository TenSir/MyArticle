# In the last version (v0.22.0), a new method mysheet.copy() (GH123) was added to xlwings.
# Could you please also add mysheet.move(), here is the api for it:
# https://stackoverflow.com/questions/64820928/move-worksheet-within-a-workbook-to-the-end


import xlwings as xw

"""
app = xw.App(visible=False, add_book=False)【速度慢】
app = xw.App(visible=False) 【速度快】
"""
app = xw.App(visible=False, add_book=False)
app.display_alerts = False
app.screen_updating = False

wb = app.books.open("1.xlsx")
rawSheet = wb.sheets[0]
print(rawSheet)
wb.save()
wb.close()
app.quit()


# import xlwings as xw
# app = xw.App(visible=False, add_book=False)
# wb = app.books.open('1.xlsx')
# sheet = wb.sheets('Sheet1')
# sheetFont_background = sheet.range('A1').api.Font.Background
# print('sheetFont_background:', sheetFont_background)
# wb.save()
# wb.close()
# app.quit()