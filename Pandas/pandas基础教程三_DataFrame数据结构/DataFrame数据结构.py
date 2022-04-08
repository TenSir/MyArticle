
# 1 创建DataFrame
# 使用字典
# import pandas as pd
# weather_data = {
#     'date':['2021-03-21','2021-03-22','2021-03-23'],
#     'temperature':[25,26,27],
#     'humidity':[81, 50, 56],
# }
# weather_df = pd.DataFrame(weather_data,columns=['date','temperature','humidity','event'])
# print(weather_df)


# import pandas as pd
# weather_data = [
#     ('2021-03-21',25,81),
#     ('2021-03-22',26,50),
#     ('2021-03-23',27,56)
# ]
# weather_df = pd.DataFrame(data = weather_data, columns=['date', 'temperature', 'humidity'])
# print(weather_df)


# import pandas as pd
# weather_data = [
#     {'date':'2021-03-21','temperature':'25','humidity':'81'},
#     {'date':'2021-03-22','temperature':'26','humidity':'50'},
#     {'date':'2021-03-23','temperature':'27','humidity':'56'}
# ]
# weather_df = pd.DataFrame(data = weather_data,
#                           columns=['date', 'temperature', 'humidity'],
#                           index = ['row_index_1','row_index_2','row_index_3']
#                           )
# print(weather_df)


