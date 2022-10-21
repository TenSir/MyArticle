# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor
#
# def fill_miss_viaRandomForest(data,missing_column):
#     for name in missing_column:
#     #for name in ["engine", "mileage", "seats"]:
#         X = data.drop(columns=f"{name}")
#         Y = data.loc[:, f"{name}"]
#         X_0 = SimpleImputer(missing_values=np.nan, strategy="constant").fit_transform(X)
#         y_train = Y[Y.notnull()]
#         y_test = Y[Y.isnull()]
#         x_train = X_0[y_train.index, :]
#         x_test = X_0[y_test.index, :]
#
#         rfc = RandomForestRegressor(n_estimators=100)
#         rfc = rfc.fit(x_train, y_train)
#         y_predict = rfc.predict(x_test)
#
#         data.loc[Y.isnull(), f"{name}"] = y_predict
#     return data
#
# df = pd.read_csv('rankingcard.csv', index_col=0)
# missing_columns = ['MonthlyIncome']
# new_df = fill_miss_viaRandomForest(df,missing_columns)
# print(new_df)


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def fill_missing_rf(X, y, fillcolumn):
    df = X.copy()
    # 待预测列
    fill_df = df.loc[:, fillcolumn]
    # 去除预测列，将剩下的列与标签组成数据集
    df = pd.concat([df.loc[:, df.columns != fillcolumn], pd.DataFrame(y)], axis=1)
    # 训练集和测试集
    y_train = fill_df[fill_df.notnull()]
    y_test = fill_df[fill_df.isnull()]
    X_train = df.iloc[y_train.index, :]
    X_test = df.iloc[y_test.index, :]
    # 预测
    rfr = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    # 对原数据进行填充并返回
    df.loc[df.loc[:, fillcolumn].isnull(), fillcolumn] = y_pred
    return df


df = pd.read_csv("rankingcard.csv")
# 不包含标签列SeriousDlqin2yrs的数据作为X
X = df.iloc[:,1:]
y = df["SeriousDlqin2yrs"]
new_df = fill_missing_rf(X,y,"MonthlyIncome")

# data.loc[:,"MonthlyIncome"].isnull().sum()


"NumberOfTime30-59DaysPastDueNotWorse"
"NumberOfTime60-89DaysPastDueNotWorse"
"NumberOfTimes90DaysLate"