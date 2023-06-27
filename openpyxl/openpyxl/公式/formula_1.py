
# import openpyxl

# from openpyxl.formula.translate import Translator
# print(openpyxl.formula)

# import openpyxl

# from openpyxl.utils import FORMULAE
# print(FORMULAE)
# print(len(FORMULAE))
#
# print('MID' in FORMULAE)
# print('min ddd' in FORMULAE)
#
#

###################################################################

# from openpyxl import load_workbook
# wbook = load_workbook(filename='formula_1.xlsx')
# wbook = load_workbook(filename='formula_1.xlsx',data_only=True)
# wsheet = wbook['Sheet1']
#
# # wsheet["C2"] = "=SUM(A2,B2)"
# # wsheet["C2"] = "=SUM(10,20)"
# # print(wsheet['C2'].value)
# for j in range(2,4):
#     cell_a = 'A' + str(j)
#     cell_b = 'B' + str(j)
#     cell_c = 'C' + str(j)
#     wsheet[cell_c] = "=SUM({},{})".format(cell_a,cell_b)
#
# cell_C2 = wsheet.cell(2,3).value
# print(cell_C2)
#
# wbook.save("formula_1.xlsx")

##################################################################

# from win32com.client import Dispatch
# xlApp = Dispatch('Excel.Application')
# xlApp.Visible = False
# xlBook = xlApp.Workbooks.Open(r'C:\Users\LEGION\Desktop\tweets_code\openpyxl\formula\formula_1.xlsx')
# xlBook.Save()
# xlBook.Close()

#################################################################

# from openpyxl import load_workbook
# # wbook = load_workbook(filename='formula_1.xlsx')
# wbook = load_workbook(filename='formula_1.xlsx',data_only=True)
#
# wsheet = wbook['Sheet1']
# cell_C2 = wsheet.cell(2,3).value
# print(cell_C2)
#
# wbook.save("formula_1.xlsx")

#################################################################

# from openpyxl import load_workbook
# from openpyxl.formula.translate import Translator
#
# wbook = load_workbook(filename='formula_1.xlsx')
# # wbook = load_workbook(filename='formula_1.xlsx',data_only=True)
#
# wsheet = wbook['Sheet1']
# # wsheet["C2"] = "=SUM(A2,B2)"
# # wsheet['F2'] = "=SUM(B2:E2)"
# # move the formula one colum to the right
# wsheet['C3'] = Translator("=SUM(A2,B2)", origin="C2").translate_formula("C3")
# print(wsheet['C3'].value)
#
# wbook.save("formula_1.xlsx")
# # wsheet['G2'].value
# # '=SUM(C2:F2)'

#################################################################

from openpyxl import load_workbook
from openpyxl.formula.translate import Translator

wbook = load_workbook(filename='formula_1.xlsx')
wsheet = wbook['Sheet1']
s = Translator("=SUM(A2,B2)", origin="C2").translate_row('9',2)

print(s)
wbook.save("formula_1.xlsx")


# cell_style = wsheet.cell(i,j).number_format
# print(cell_style)