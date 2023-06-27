# from openpyxl import *
# from openpyxl.utils import get_column_letter
# #from openpyxl.styles.styleable import StyleableObject
#
# from openpyxl import *
#
#
# def auto_format_cell_width(ws):
#     for letter in range(1,ws.max_column):
#         maximum_value = 0
#         for cell in ws[get_column_letter(letter)]:
#             val_to_check = len(str(cell.value))
#             if val_to_check > maximum_value:
#                maximum_value = val_to_check
#         ws.column_dimensions[get_column_letter(letter)].width = maximum_value + 1
#
#
# def adjust_column_dimension(ws, min_row, min_col, max_col):
#     column_widths = []
#     for i, col in  enumerate(ws.iter_cols(min_col=min_col, max_col=max_col, min_row=min_row)):
#         for cell in col:
#             value = cell.value
#             if value is not None:
#                 if isinstance(value, str) is False:
#                     value = str(value)
#                 try:
#                     column_widths[i] = max(column_widths[i], len(value))
#                 except IndexError:
#                     column_widths.append(len(value))
#     for i, width in enumerate(column_widths):
#         col_name = get_column_letter(min_col + i)
#         value = column_widths[i] + 2
#         ws.column_dimensions[col_name].width = value
# # 使用 stackoverflow.com
# # adjust_column_dimension(ws, 1,1, ws.max_column)
#
#
# wb = load_workbook('cell_operation.xlsx')
# ws = wb['Sheet1']
#################################################################
# cell_value_1 = ws.cell(column=1, row=1).value
# set_value_1 = ws.cell(column=1, row=9).value = 8
# set_value_2 = ws.cell(column=1, row=10).value = '9'
#
# ws.cell(column=2, row=9, value="{0}".format(get_column_letter(1)))
#
# print(ws.cell(column=2, row=9).column_letter)
# print(ws.cell(column=2, row=9).coordinate)
# print(ws.cell(column=2, row=9).col_idx)
# print(ws.cell(column=2, row=9).encoding)
# print(ws.cell(column=2, row=9).offset)
# print(ws.cell(column=2, row=9).is_date)
# print(ws.cell(column=2, row=9).data_type)

# ws.cell(3,3).value = 'python知识学堂'

# from openpyxl.cell import read_only
# s_value = read_only.ReadOnlyCell( 'Sheet1', row=2, column=2, value=3, data_type='n')
# print(s_value.value)
# print(s_value.internal_value)

# from openpyxl.cell import text
# print(text.InlineFont().shadow)

# 调整列宽
# ws.column_dimensions['A'].width = 20.0
# 调整行高
# ws.row_dimensions[1].height = 40

# ws.merge_cells("A1:B1")
# ws.merge_cells(start_column=3,end_column=5,start_row=3,end_row=5)
# print(ws.merged_cells)
# # A1:B1 C3:E5
# print(ws.merged_cell_ranges)
# # [<MergedCellRange A1:B1>, <MergedCellRange C3:E5>]
# wb.save('cell_operation.xlsx')
# wb.close()
#
# letter = chr(i + 65)  # 由ASCII值获得对应的列字母

#cell_1 = ws.cell(column=1, row=1, value="{0}".format(get_column_letter(4)))


