import pandas as pd
import numpy as np

iterables = [['1', '2', '3'], ['bule', 'green']]
arrays = pd.MultiIndex.from_product(iterables, names=['number', 'color'])
use_df = pd.DataFrame(
    np.random.randn(6,6),
    index = arrays,
    columns=arrays
)

print(use_df)
print('________________________________________________')
# print(use_df.index)
# print(use_df.index.values)
# print(use_df.columns.values)


# print(use_df.index.get_level_values(0))
# print(use_df.index.get_level_values('number'))
# print(use_df.index.get_level_values(1))
# print(use_df.index.get_level_values('color'))

# # 获取标签或标签元组的位置
# print(use_df.index.get_loc('1'))
# # 获取标签序列的位置
# print(use_df.index.get_locs('1'))
# print(use_df.index.get_locs('2'))


# print(use_df['1'].columns.values)
# print(use_df['2'].columns.values)
# print(use_df['1'].index.values)
# print(use_df['2'].index.values)
# print(use_df[['2','3']].index.values)

# index = pd.MultiIndex.from_product(
#     [['1', 'bule'], ['3', 'green']],
#     names=['number', 'color'])
# print(index)
# print(index.to_flat_index())


# print(use_df['1'])
# # print(use_df['1'])
# #
# # # 列优先取数
# # print(use_df.loc['1'])
#
#
# print(use_df.loc[:]['1'])



print('____________________________')
print(use_df.iloc[1])
print('____________________________')
print(use_df.iloc[1]['1'])
print('____________________________')
print(use_df.iloc[1][1])