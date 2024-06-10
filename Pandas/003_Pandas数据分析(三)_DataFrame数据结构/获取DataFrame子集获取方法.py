import pandas as pd
data = [
         [1,2,3,4,5],
         [6,7,8,9,10],
         [11,11,12,13,14],
         [15,16,17,18,19],
         [20,21,22,23,24],
         [25,26,27,28,29],
         [30,31,32,33,34],
         [35,36,37,38,39],
         [40,41,42,43,44],
         [45,46,47,48,49]
      ]
index = ['01', '02', '03', '04', '05','06', '07', '08', '09', '10']
column=['C001', 'C002', 'C003','C004','C005']

df_example = pd.DataFrame(data= data,index=index,columns=column)
print(df_example)
print('____________________________')

# print(type(df_example[['C001']]))
# print(type(df_example['C001']))
# print(df_example.loc['01'])


# print(df_example.head())
# print(df_example.tail())

# print(df_example[1:5:2])
# print(df_example[1:5])


# print(df_example['C001'])
# print(df_example[['C001','C002']])
#
# print(type(df_example['C001']))
# print(type(df_example[['C001']]))


# print(df_example.loc['01'])
# print(df_example.loc[['01','02','03']])
#
# print(df_example.iloc[0])
# print(df_example.iloc[[0,1,2]])


"""
df.loc[lambda df: df['shield'] == 8]
                    max_speed  shield
        sidewinder          7       8
"""


# print(df_example.loc[[False,True,False,True,False,True,False,True,False,True]])

# 进行转置
# print(df_example.T)

# ndarrary
print(df_example.values[1])
print(type(df_example.values))