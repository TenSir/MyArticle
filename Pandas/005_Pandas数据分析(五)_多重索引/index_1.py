# # import pandas as pd
# # import numpy as np
# # #下面显示构造pd.MultiIndex
# # df1=pd.DataFrame(np.random.randint(0,150,size=(6,3)),columns=['java','html5','python'])
#
# # 创建dataframe
# # 1.ndarray数组
# import pandas as pd
# import numpy as np
# pd_test1 = pd.DataFrame(
#         np.random.randint(0,150,size=(4,3)),
#         columns=['java','html5','python'],
#         index=pd.MultiIndex.from_arrays
#         (
#             [
#                 ['a','b','c','d'],
#                 ['A','B','C','D']
#             ]
#         )
#     )
# # print(pd_test1)
# # print(pd_test1.index)
#
# # 2.pd.MultiIndex.from_frame
# pd_test2 = pd.DataFrame(
#         np.random.randint(0,150,size=(4,3)),
#         columns=['java','html5','python'],
#     )
#
# # print(pd_test2)
# # print(pd.MultiIndex.from_frame(pd_test2))
# # print(pd.MultiIndex.from_frame(pd_test2,names=['J','H','P']))
#
#
# #3.pd.MultiIndex.from_tuples
# arr_test3 = [(1, 'red'),(1, 'blue'),
#             (2, 'red'), (2, 'blue'),
#             (3, 'red'), (3, 'blue')]
# index = pd.MultiIndex.from_tuples(arr_test3, names=('number', 'color'))
# series_test3 = pd.Series(arr_test3, index=index)
# pd_test3 = pd.DataFrame(arr_test3, index=index)
#
# # print(pd_test3)
# # print(series_test3)
# # print(pd_test3.index)
#
#
# # 4.pd.MultiIndex.from_product
# number = [1,2,3]
# color = ['green', 'purple','blue']
#
# index = pd.MultiIndex.from_product([number,color], names=['number', 'color'])
# pd_test4 = pd.DataFrame(np.random.randn(9), index=index)
#
# print('111111111111111111111111111')
# print(pd_test4)
# print(pd_test4.index)





import pandas as pd
import numpy as np
pd_test1 = pd.DataFrame(
        np.random.randint(0,150,size=(4,3)),
        columns=['java','html5','python'],
        index=pd.MultiIndex.from_arrays
        (
            [
                ['a','b','c','d'],
                ['A','B','C','D']
            ]
        )
    )

print(pd_test1)



multi_index = pd.MultiIndex.from_product([list('ABCD'), ['Female','Male']], names=('School', 'Gender'))

multi_column = pd.MultiIndex.from_product([['Height', 'Weight'], ['Freshman','Senior','Sophomore','Junior']], names=('Indicator', 'Grade'))

df_multi = pd.DataFrame(np.c_[(np.random.randn(8,4)*5 + 163).tolist(), (np.random.randn(8,4)*5 + 65).tolist()],
                        index = multi_index, columns = multi_column).round(1)


df_multi.index.get_level_values














