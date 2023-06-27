
#
# from openpyxl.utils.dataframe import dataframe_to_rows
#
# wb = Workbook()
# ws = wb.active
#
# for r in dataframe_to_rows(df, index=True, header=True):
#     ws.append(r)
#


# https://openpyxl.readthedocs.io/en/stable/pandas.html#numpy-support


import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows


wbook = load_workbook(filename='np_pd_test.xlsx')
wsheet = wbook['Sheet1']

data = {'alpha': ['A', 'B', 'C', 'D', 'E', 'F'],
        'num_1': [25, 32, 18, np.nan, 14, 15],
        'num_2': [12, 15, 17, 18, 22, 23],
        }
labels = ['a', 'b', 'c', 'd', 'e', 'f']
df = pd.DataFrame(data, index=labels)

for each in dataframe_to_rows(df, index=False, header=True):
    wsheet.append(each)

for cell in wsheet['A'] + wsheet[1]:
    cell.style = 'Pandas'

wbook.save("np_pd_test.xlsx")
