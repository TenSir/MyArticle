import numpy as np
import pandas as pd

from openpyxl.cell.cell import WriteOnlyCell
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

data = {'alpha': ['A', 'B', 'C', 'D', 'E', 'F'],
        'num_1': [25, 32, 18, np.nan, 14, 15],
        'num_2': [12, 15, 17, 18, 22, 23],
        }
labels = ['a', 'b', 'c', 'd', 'e', 'f']
df = pd.DataFrame(data, index=labels)

wb = Workbook(write_only=True)
ws = wb.create_sheet()

cell = WriteOnlyCell(ws)
cell.style = 'Pandas'

def format_first_row(row, cell):
    for c in row:
        cell.value = c
        yield cell

rows = dataframe_to_rows(df)
first_row = format_first_row(next(rows), cell)
ws.append(first_row)

# for row in rows:
#     row = list(row)
#     cell.value = row[0]
#     row[0] = cell
#     ws.append(row)

wb.save("openpyxl_stream.xlsx")