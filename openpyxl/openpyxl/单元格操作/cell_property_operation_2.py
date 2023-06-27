from openpyxl import*

# from openpyxl.styles import Font
# from openpyxl.styles import Border,Side
# wbook = load_workbook("cell_property_sets.xlsx")#使用openpyxl读取xlsx文件，创建workbook
# wsheet = wbook['Sheet1']
# ws = wb.active

# 字体名称
# 字体大小
# 字体颜色
# 加粗
# 斜体
# 垂直对齐方式
# 下划线
# 删除线
#######################################################################
from openpyxl import*
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
wbook = load_workbook("cell_property_sets.xlsx")
wsheet = wbook['Sheet1']

#######################################################################
# font = Font(name='微软雅黑',
#             size=11,
#             color='FF000000',
#             bold=True,
#             italic=True,
#             vertAlign='baseline',
#             underline='double',
#             strike=False)
#
# wsheet['A2'].font = font
#######################################################################
# side_type = Side(border_style='mediumDashDot',color='FF000000')
# border = Border(left=side_type,
#                 right=side_type,
#                 top=side_type,
#                 bottom=side_type,
#                 diagonal=side_type,
#                 diagonal_direction=30,
#                 outline=side_type,
#                 vertical=side_type,
#                 horizontal=side_type
#                 )
# wsheet['A3'].border = border
#######################################################################
# fill = PatternFill(fill_type = 'darkDown',start_color='A6DA70D6',end_color='000000')
# wsheet['A4'].fill = fill
#######################################################################
# horizontal为水平方向，vertical为竖直方向，
# align = Alignment(horizontal='center',vertical='center',text_rotation=0,wrap_text=True,shrink_to_fit=True,indent=0)
# wsheet['A6'].alignment = align
#######################################################################



wsheet['A9'].number_format = 'd-mmm-yy'
wbook.save("cell_property_sets.xlsx")
wbook.close()

# 参数说明:
# Horizontal:水平方向，左对齐left，居中center对齐和右对齐right可选。
# Vertical:垂直方向，有居中center，靠上top，靠下bottom，两端对齐justify等可选。
# text_rotation:文本旋转度。
# wrap_text:自动换行
# Indent:缩进。

#
# from openpyxl.styles import Font
#
# fsheet1 = Font(name='Arial', size=10)
# # 复制时指定字体为“微软雅黑”，其他属性来自fsheet1
# fsheet2 = fsheet1.copy(name="微软雅黑")

