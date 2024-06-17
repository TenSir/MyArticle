# import pandas as pd
# print(pd.Index)
# print(pd.Index.__doc__)



# 获取索引对象
# import pandas as pd
# df_Test = pd.DataFrame(
#     {'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},
#     #index=['a','b','c']
#     index=[1,2,3]
# )
# print(df_Test)
# print('获取的索引对象为：',df_Test.index)
# print('获取的索引对象值为：',df_Test.index.values)


# import pandas as pd
# inx_1 = pd.Index(['a','b','c'])
# inx_2 = pd.Index([1,2,3])
# inx_3 = pd.Index([1,2,'3'])
# # print(inx_1)
# # print(inx_2)
# # print(inx_3)
#
# print('____________________________________________________')

# 索引的可重复性
# import pandas as pd
# df_Test_2 = pd.DataFrame(
#     {'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},
#     index=['a','b','c']
# )
# print(df_Test_2)
# print(df_Test_2.index.is_unique)
# print(df_Test_2.index.isna())
#
# exit(0)
# print(df_Test_2.reset_index(drop=True))
# print('____________________________________________________')


# import pandas as pd
# df_Test = pd.DataFrame(
#     {'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},
#     #index=['a','b','c']
#     index=[1,2,3])
#
# N_df_Test = df_Test.set_index(['time'],drop=True)
# print(N_df_Test)
#
# new_index = N_df_Test.index.rename('这是索引哦')
# N_df_Test.index=new_index
# print(N_df_Test)
# print('____________________________________________________')

# 还原reset_index
# import pandas as pd
# df_Test = pd.DataFrame(
#     {'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},
#     index=list('abc'))
# print(df_Test)
# print(df_Test.reset_index())
# print(df_Test.reset_index(drop=True))

# print('____________________________________________________')

# reindex
# import pandas as pd
# df_Test_3 = pd.DataFrame(
#     {'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},
#     index = [0,1,2]
# )
#
# print(df_Test_3)
# print(df_Test_3.reindex([1,2,3,4,5]))
# print(df_Test_3.reindex(['a','b',5]))
#
# print(df_Test_3.reindex([1,2,3,4,5],method='nearest'))
#
# print(df_Test_3.reindex(columns=['time','New_count']))
#
# print(df_Test_3.reindex(columns=['time','New_count']))
# print(df_Test_3.reindex(index = [0,2],columns=['time','count']))


#
# # print(df_Test_3.reindex([1,2,3],method='pad'))
# # print(df_Test_3.reindex([1,2,3],method='ffill'))
#
# df_Test_3.index=['a','b',5]
# print(df_Test_3)
# print('____________________________________________________')



import pandas as pd
# 创建两个索引对象
index_1 = pd.Index(['a','a','b','c','d',1,2,3,4,5,6])
index_2 = pd.Index([1,2,3,4,5,6])
# 创建一个series
series_1 = pd.Series(range(11),index=index_1)

# 判断目标索引是否存在
print('a' in index_1)
print(1 in index_1)

# 长度
print(len(index_1))

# 切片操作,跟列表的切片是一样的
print(index_1[:3])

# 两个索引对象的交集
print(index_1.intersection(index_2))

# 两个索引对象的差集
print(index_1.difference(index_2))
print(index_2.difference(index_1))

# 索引对象的连接
print(index_1.append(index_2))

# 删除索引(传入列表，列表中的值为删除的位置)
print(index_1.delete([0]))
print(index_1.delete([0,1,3]))

# 插入
print(index_1.insert(0,'aaa'))

# 去重
print(index_1.is_unique)
print(index_1.unique())
