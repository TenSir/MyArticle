
import openpyxl.drawing.spreadsheet_drawing
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference, Series

#1.########################################################
# workbook = Workbook()
# worksheet = workbook.active
# for i in range(10):
#      worksheet.append([i])
#
# values = Reference(worksheet, min_col=1, min_row=1, max_col=1, max_row=10)
# chart = BarChart()
# chart.height = 5
# chart.width = 10
# chart.anchor = 'C1'
# chart.add_data(values)
# worksheet.add_chart(chart)
# workbook.save("TestChart.xlsx")



#2.########################################################

from openpyxl import Workbook
from openpyxl.chart import ScatterChart,Reference,Series

workbook = Workbook()
worksheet = workbook.active

worksheet.append(['X', '1/X'])
for x in range(-10, 11):
    if x:
        worksheet.append([x, 1.0 / x])

chart1 = ScatterChart()
chart1.title = "Full Axes"
chart1.x_axis.title = 'x'
chart1.y_axis.title = '1/x'
chart1.legend = None

chart2 = ScatterChart()
chart2.title = "Clipped Axes"
chart2.x_axis.title = 'x'
chart2.y_axis.title = '1/x'
chart2.legend = None

chart2.x_axis.scaling.min = 0
chart2.y_axis.scaling.min = 0
chart2.x_axis.scaling.max = 11
chart2.y_axis.scaling.max = 1.5

x = Reference(worksheet, min_col=1, min_row=2, max_row=22)
y = Reference(worksheet, min_col=2, min_row=2, max_row=22)
s = Series(y, xvalues=x)
chart1.append(s)
chart2.append(s)

worksheet.add_chart(chart1, "C1")
worksheet.add_chart(chart2, "C15")

workbook.save("minmax.xlsx")

chart.x_axis.scaling.orientation = "minMax"
chart.y_axis.scaling.orientation = "maxMin"