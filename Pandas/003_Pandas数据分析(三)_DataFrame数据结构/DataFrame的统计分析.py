import numpy as np
import pandas as pd

df = pd.DataFrame([[1.2,-2.5,0.3],[-4.2,1.5,np.nan],
                   [np.nan,np.nan,np.nan],[-0.5,2.3,0.6]],
                   index = ['A','B','C','D'],
                   columns= ['num_1','num_2','num_3'])

print(df)
print(df.sum(axis=1))
# print(df.sum(axis=0))
# print(df.describe())
# print(df.describe())

# print(df.median())
# print(df.sum()['num_1'])

# print(df.min(axis = 1))
# print(df.min(axis = 0))

# print(df.num_1)
# print(df.num_1.argmin())
# print(df.loc['A'])
# print(df.loc['A'].argmax())

# print(df.num_1.idxmin())
# print(df.loc['A'].idxmax())

print(df.cummin(axis = 0,skipna=True))
# print(df.cummin(axis = 1))
# print(df.num_1.cummin(axis = 0))


