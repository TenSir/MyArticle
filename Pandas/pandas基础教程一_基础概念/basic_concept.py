import pandas as pd
test_list = [1,2,3,4,5]
t_series = pd.Series(test_list)

data = {
    'name':['Beijing','Shanghai','Xiamen','Wuhan','Chizhou'],
    'year':[2000,2001,2002,2001,2002],
    'stage':[1,0,1,0,0]
}
t_dataframe = pd.DataFrame(data)


print(t_series)
print(t_dataframe)
