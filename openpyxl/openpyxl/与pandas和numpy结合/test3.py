import pandas as pd
from itertools import islice
from openpyxl import load_workbook

wbook = load_workbook('np_pd_test.xlsx')
wsheet = wbook['Sheet1']

data = wsheet.values
print('data:',data)

cols = next(data)[1:]
data = list(data)
idx = [r[0] for r in data]
data = (islice(r, 1, None) for r in data)
df = pd.DataFrame(data, index=idx, columns=cols)

print(cols)
print(df)



