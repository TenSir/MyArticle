

#1.Series创建
# import pandas as pd
# import pandas as pd
# lst = [1,2,3,4,5,6]
# sl = pd.Series(lst)
#
# import pandas as pd
# dst = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
# sd = pd.Series(dst)
# print(sd)
# print(sd.values)
#
# import pandas as pd
# # obj = pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f'])
# obj = pd.Series([1,2,3,4,5,6],index=['a','b','c','d','e','f'])
# loc
# print(obj['a'])
# print(obj['a':'d'])
# print(obj.loc['a'])
# print(obj.loc['a':'d'])
# print(obj.loc[['a','d']])


# iloc
# print(obj.iloc[0])
# print(obj.iloc[0:2])
# print(obj.iloc[[0,2]])

# print(obj.size)
# print(obj.shape)
# print(obj.dtype)
# print(obj.values)

# #2.series常用的方法
# import pandas as pd
# obj1 = pd.Series([1,2,3,4,5,6,7,7],index=['a','b','c','d','e','f','g','h'])
# obj2 = pd.Series([111,222,333],index=['x','y','z'])
# obj3 = pd.Series([2,3,4],index=['x','y','z'])
# # print(obj.min())
# # print(obj.max())
# print(obj1.describe())
# print(obj1.sample())
# print(obj1.unique())
# print(obj1.append(obj2))
# print(obj2.equals(obj3))
#
# # 两种不同的in
# print(obj1.isin(['1','2']))
# print('a' in obj1)
# print(obj1.isnull())
# print(obj1.drop_duplicates())


#3.其他的操作

import pandas as pd
import numpy as np
obj1 = pd.Series([1,2,3,4,5,6,7,7],index=['a','b','c','d','e','f','g','h'])
print(obj1[obj1 > obj1.mean()])












