

"""
from openpyxl import Workbook
from openpyxl.comments import Comment
wb = Workbook()
ws = wb.active
comment = Comment('这是一个批注', 'Python知识学堂')
ws["A1"].value = '测试文本'

ws["A1"].comment = comment
wb.save('commented_book.xlsx')
# comment.text
# 'This is the comment text'
# comment.author

"""


# from openpyxl.workbook import Workbook
# from openpyxl.worksheet.properties import, PageSetupProperties
# wb = Workbook()
# ws = wb.active
# wsprops = ws.sheet_properties
# wsprops.tabColor = "DA70D6"
# wsprops.filterMode = True
# wsprops.pageSetUpPr = PageSetupProperties(fitToPage=True, autoPageBreaks=False)
# wsprops.outlinePr.summaryBelow = False
# wsprops.outlinePr.applyStyles = True
# wsprops.pageSetUpPr.autoPageBreaks = True
# wb.save('sheet_propertie_2.xlsx')


from openpyxl import Workbook
from openpyxl.comments import Comment

wbook=Workbook()
wsheet=wbook.active
wsheet["A1"].value = '人生苦短，我用Python'
comment = Comment("这是一个comment", "Python知识学堂")
comment.width = 300
comment.height = 50

wsheet["A1"].comment = comment

wbook.save('commented_test.xlsx')