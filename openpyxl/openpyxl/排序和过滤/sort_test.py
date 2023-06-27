

########################################################################
# from openpyxl import *
# workbook = load_workbook('sort_file.xlsx')
# worksheet = workbook['Sheet1']
#
# worksheet.auto_filter.ref = "A1:B12"
# worksheet.auto_filter.add_filter_column(0, ["Kiwi", "Apple", "Mango"])
# worksheet.auto_filter.add_sort_condition("B2:B12")
#
# workbook.save("sort_file.xlsx")

########################################################################


import pandas as pd
data_test = pd.read_excel('sort_file.xlsx')
df = pd.DataFrame(data_test)
# 以列“Fruit”的标签列来进行升序排列
df_1 = df.sort_values('Fruit', ascending=True)
writer = pd.ExcelWriter('sort_file.xlsx')
df_1.to_excel(writer, sheet_name ='Sheet1', index=False)
writer.save()


# df_2 = df.sort_values(['Fruit','Price'],ascending=False)
# df_3 = df.sort_values(['Fruit','Price'],ascending=[False,True])

