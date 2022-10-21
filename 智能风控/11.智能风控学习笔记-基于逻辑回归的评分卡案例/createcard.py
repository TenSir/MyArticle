
import scipy
import toad
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression as LR
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('rankingcard.csv', index_col=0)
print(df.shape)
df.info()



df.drop_duplicates(inplace=True)
df.info()


df.index=range(df.shape[0])

# 重置索引
df.reset_index(drop=True, inplace=True)

# 检查各个特征的缺失情况
df.isnull().sum()/df.shape[0]

# 填充家庭成员缺失值 ，将家庭成员缺少值全部填充为0
df["NumberOfDependents"].fillna(df["NumberOfDependents"].mean(),inplace=True)

df = df[df['age']>0]
df = df[df['NumberOfTime30-59DaysPastDueNotWorse']<90]
df = df[df['NumberOfTime60-89DaysPastDueNotWorse']<90]
df = df[df['NumberOfTimes90DaysLate']<90]
df.reset_index(drop=True, inplace=True)
df.info()

#探索样本分布
df['SeriousDlqin2yrs'].value_counts()


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=20,n_jobs=-1)
# 返回已经上采样后的数据和标签
X_old = df.iloc[:,1:]
y_old = df["SeriousDlqin2yrs"]
X,y = sm.fit_resample(X_old,y_old)
# 结果转为dataframe
X = pd.DataFrame(X)
y = pd.DataFrame(y)
y.value_counts()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=20)
train_data = pd.concat([y_train,X_train], axis=1)
train_data.reset_index(drop=True, inplace=True)
test_data = pd.concat([y_test, X_test], axis=1)
test_data.reset_index(drop=True, inplace=True)
train_data.to_csv('train_data.csv',index = False)
test_data.to_csv('test_data.csv',index = False)




import scipy
def graphforbestbin(df, X, y, n, q=20, graph=True):
    '''
    基于卡方检验的分箱
    df: 需要输入的数据
    X: 需要分箱的列名
    y: 分箱数据对应的标签 y列名
    n: 保留分箱个数
    q: 初始分箱的个数
    graph: 是否要画出IV图像
    '''
    df = df[[X, y]].copy()
    # 调用pandas的分箱函数
    df["qcut"], bins = pd.qcut(df[X], retbins=True, q=q, duplicates="drop")
    coount_y0 = df.loc[df[y] == 0].groupby(by="qcut").count()[y]
    coount_y1 = df.loc[df[y] == 1].groupby(by="qcut").count()[y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]

    # 判断每个箱子是否包含正负样本
    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2] + num_bins[1][2],
                num_bins[0][3] + num_bins[1][3])]
            continue

        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i - 1:i + 1] = [(
                    num_bins[i - 1][0],
                    num_bins[i][1],
                    num_bins[i - 1][2] + num_bins[i][2],
                    num_bins[i - 1][3] + num_bins[i][3])]
                break
        else:
            break

    # 定义WOE函数
    def get_woe(num_bins):
        columns = ["min", "max", "count_0", "count_1"]
        df = pd.DataFrame(num_bins, columns=columns)
        df["total"] = df.count_0 + df.count_1
        df["percentage"] = df.total / df.total.sum()
        df["bad_rate"] = df.count_1 / df.total
        df["good%"] = df.count_0 / df.count_0.sum()
        df["bad%"] = df.count_1 / df.count_1.sum()
        df["woe"] = np.log(df["good%"] / df["bad%"])
        return df

    # 定义IV函数
    def get_iv(df):
        rate = df["good%"] - df["bad%"]
        iv = np.sum(rate * df.woe)
        return iv

    # 找最合理的分箱数n
    bins_df = None
    IV = []
    axisx = []
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins) - 1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i + 1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            pvs.append(pv)

        i = pvs.index(max(pvs))
        num_bins[i:i + 2] = [(
            num_bins[i][0],
            num_bins[i + 1][1],
            num_bins[i][2] + num_bins[i + 1][2],
            num_bins[i][3] + num_bins[i + 1][3])]

        bins_df = pd.DataFrame(get_woe(num_bins))
        axisx.append(len(num_bins))
        IV.append(get_iv(bins_df))

    if graph:
        plt.figure()
        plt.plot(axisx, IV)
        plt.xticks(axisx)
        plt.xlabel(f"number of box - {X}")
        plt.ylabel("IV")
        plt.show()
    return bins_df


# 对训练集查看分箱的最优数
col = [each for each in train_data.columns if each not in [ 'SeriousDlqin2yrs','NumberOfDependents']]
for columns in col:
    graphforbestbin(train_data,columns,"SeriousDlqin2yrs",n=1,q=20,graph=True)


# 确定的分箱自动处理
sure_bins = {"RevolvingUtilizationOfUnsecuredLines":5,
            "age":5,
            "DebtRatio":4,
            "MonthlyIncome":3,
            "NumberOfOpenCreditLinesAndLoans":5
           }
# 不确定的分箱单独处理
unsure_bin = {"NumberOfTime30-59DaysPastDueNotWorse":[0,1,2,13],
              "NumberOfTimes90DaysLate":[0,1,2,17],
              "NumberRealEstateLoansOrLines":[0,1,2,54],
              "NumberOfTime60-89DaysPastDueNotWorse":[0,1,2,9],
              "NumberOfDependents":[0,1,2,3]
              }
# 设置分箱区间区间结果使用 np.inf替换最大值，用-np.inf替换最小值
hand_bins = {k:[-np.inf,*v[:-1],np.inf] for k,v in unsure_bin.items()}


# 训练数据集开始分箱
# 字典存储分箱结果
bin_of_col_train={}
# 生成确定分箱的分箱区间和分箱后的 IV 值
for col in sure_bins:
    bins_df=graphforbestbin(train_data,
                            col,
                            'SeriousDlqin2yrs',
                            n=sure_bins[col],
                            q=20,
                            graph=False)
    bins_list=sorted(set(bins_df["min"]).union(bins_df["max"]))
    #保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
    bins_list[0],bins_list[-1]=-np.inf,np.inf
    bin_of_col_train[col]=bins_list
#合并手动分箱数据
bin_of_col_train.update(hand_bins)
bin_of_col_train


# 获取WOE值
def get_woe(df,col,y,bins):
    df=df[[col,y]].copy()
    df["cut"]=pd.cut(df[col],bins)
    bins_df=df.groupby("cut")[y].value_counts().unstack()
    woe=bins_df["woe"]=np.log((bins_df[0]/bins_df[0].sum())/(bins_df[1]/bins_df[1].sum()))
    return woe
#将所有特征的WOE存储到字典当中
woe={}
for col in bin_of_col_train:
    woe[col]=get_woe(train_data,col,"SeriousDlqin2yrs",bin_of_col_train[col])
woe


# 创建WOE dataframe
train_data_woe=pd.DataFrame(index=train_data.index)
# 对所有特征进行映射：
for col in bin_of_col_train:
    train_data_woe[col]=pd.cut(train_data[col],bin_of_col_train[col]).map(woe[col])
# 添加标签
train_data_woe["SeriousDlqin2yrs"]=train_data["SeriousDlqin2yrs"]
train_data_woe.head()


# 测试集处理
test_data_woe = pd.DataFrame(index=test_data.index)
# 训练集和测试集同一个分箱列
for col in bin_of_col_train:
    test_data_woe[col] = pd.cut(test_data[col],bin_of_col_train[col]).map(woe[col])
test_data_woe["SeriousDlqin2yrs"] = test_data["SeriousDlqin2yrs"]
test_data_woe.head()


train_data_woe.to_csv('train_data_woe.csv',index = False)
test_data_woe.to_csv('test_data_woe.csv',index = False)


# 模型输入的数据集构建
col = [each for each in train_data_woe.columns if each != "SeriousDlqin2yrs" ]
X_train = train_data_woe[col]
y_train=train_data_woe['SeriousDlqin2yrs']
X_test=test_data_woe[col]
y_test=test_data_woe['SeriousDlqin2yrs']
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train, y_train)
# 查看得分
print(lr.score(X_test, y_test))


c_1 = np.linspace(0.01,0.2,20)
score = []
for i in c_1:
    lr = LR(solver='liblinear',C=i).fit(X_train,y_train)
    score.append(lr.score(X_test,y_test))
plt.figure()
plt.plot(c_1,score)
plt.show()



score = []
find_list = list(range(10))
for i in find_list:
    lr = LR(solver='liblinear',C=0.025,max_iter=i).fit(X_train,y_train)
    score.append(lr.score(X_test,y_test))
plt.figure()
plt.plot(find_list,score)
plt.xlabel("iter num")
plt.ylabel("score")
plt.show()


# 模型评估
import scikitplot as skplt
vali_proba_df = pd.DataFrame(lr.predict_proba(X_test))
skplt.metrics.plot_roc(y_test,
                       vali_proba_df,
                       plot_micro=False,
                       figsize=(5,5),
                       plot_macro=False)




#缺失率、IV、相关系数进行特征筛选
selected_data, drop_list=toad.selection.select(
                   train_data,
                   train_data["SeriousDlqin2yrs"],
                   empty=0.7,
                   iv=0.03,
                   corr=1,
                   return_drop=True,
                   exclude=["SeriousDlqin2yrs"])

B = 20/np.log(2)
A = 600 + B*np.log(1/60)

base_score = A - B*lr.intercept_
base_score

score_RevolvingUtilizationOfUnsecuredLines = woe["RevolvingUtilizationOfUnsecuredLines"] * (-B*lr.coef_[0][0])
print(score_RevolvingUtilizationOfUnsecuredLines)

# 写入所有特征的评分卡内容
file = "ScoreRes.csv"
with open(file,"w") as f:
    f.write("base_score,{}\n".format(base_score))
for i,col in enumerate(train_data.columns):
    score = woe[col] * (-B*lr.coef_[0][i])
    score.name = "Score"
    score.index.name = col
    score.to_csv(file,header=True,mode="a")


# 保存训练完结束的模型
import joblib
joblib.dump(lr, "ScoreCard.pkl")

# 模型加载和预测
# 加载和预测
new_lr = joblib.load("ScoreCard.pkl")

# proba = new_lr.predict_proba(X_test.loc[0, :].values.reshape(1, -1))
# proba = new_lr.predict_proba([X_test.loc[0, :]])
proba = new_lr.predict_proba(pd.DataFrame([X_test.loc[0, :]]))
proba_class = new_lr.predict(pd.DataFrame([X_test.loc[0, :]]))
# 预测值
print(proba)
# 预测所属类别
print(proba_class)