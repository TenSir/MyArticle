# 导入相关库并初始化设置
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, \
    precision_score, recall_score, roc_curve, roc_auc_score, precision_recall_curve  # 导入指标库
import matplotlib.pyplot as plt
import prettytable  # 导入表格库
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import toad  
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
# 风格设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文
sns.set(style="ticks") # 设置风格


# ---------------------------------------------------------变量处理---------------------------------------------------------

# 变量类型判定
def var_class(df, target, date_col=None, ncount=10):
    '''
    获取变量的类型
    
    df:数据表
    target:y变量
    date_col:日期变量，不提供则自动判断
    ncount:分类变量的取值数，默认<=10
    '''
    df1 = df.drop(target, axis=1)
    df2 = df1.apply(pd.to_numeric, errors='ignore') # 把能转换成数字的都转换成数字，不能转化的由error=True参数控制忽略掉。
    df3 = df2.select_dtypes(object).apply(pd.to_datetime, errors='ignore') # 此处注意，要把先选出object类别数据。毕竟pd.to_datetime()也可以将阿拉伯数字变成时间格式。

    dic_init={}
    # 筛选y变量
    dic_init[target]='y'
    # 筛选日期变量
    date_col = date_col
    if date_col:
        for d in date_col:
            dic_init[d]='date'
    else:
        for i,v in df3.dtypes.items():
            if v=='datetime64[ns]':
                dic_init[i]='date'
    # 筛选连续变量
    for i in df2.select_dtypes(include='number').columns:
        if df2[i].nunique()>ncount:
            dic_init[i]='number'

    # 筛选分类变量
    sur_col=[x for x in df.columns if x not in dic_init.keys()]
    dic_init.update(dict(zip(sur_col,['object']*len(sur_col))))
    
    return dic_init

# 获取各类型变量
def get_key(dic, value):
    '''
    获取特定值的键
    dic:字典
    value:特定值
    '''
    return list(filter(lambda k:dic[k] == value, dic))

# 字符串类型变量标签化
def obj_label(df, ex_lis=[]):
    '''
    df:数据集
    ex_lis：不需要标签化的列lisit
    '''
    df=df.copy()
    x_obj = [x for x in df.select_dtypes(object).columns if x not in ex_lis]
    df[x_obj] = df[x_obj].astype('str') 
    dl = defaultdict(LabelEncoder)
    df[x_obj]=df[x_obj].apply(lambda x: dl[x.name].fit_transform(x))
    
    return df

# 连续变量分箱
def number_col_bins(df, nums_col, y, method='chi'):
    '''
    df:数据集
    nums_col:连续变量list
    y:y变量
    method:分箱方式
    
    return:分箱后的df
    '''
    df=df[nums_col+[y]].copy()
    # 连续变量分箱
    combiner = toad.transform.Combiner()  
    combiner.fit(df, df[y], method=method, # 卡方分箱
                    min_samples=0.05) 
    # 导出箱的节点  
    vars_bins = combiner.export() 
    # 分箱
    for x in nums_col:
        xmax=df[x].max()
        var_bins = []
        var_bins.append(float('-inf'))
        var_bins.extend(vars_bins[x])
        var_bins.append(xmax)
        df[x]=pd.cut(df[x], bins=var_bins, labels=var_bins[1:])
    return df

# 单变量eda
def var_eda(df, x, y, fonts=12):
    '''
    df:数据集
    x:变量x
    y:变量y
    object_col：分类变量列表
    '''

    #第一个图，柱状图
    ax1 = sns.countplot(x=x, data = df, color=sns.xkcd_rgb['windows blue'])
    ax1.set_xlabel(x, fontsize=fonts)
    ax1.set_ylabel(None)
    # sns.despine(ax=ax1) # 剔除右上边框
    #第二个图，点线图
    ax2 = ax1.twinx() # 共享x轴
    ax2 = sns.pointplot(x=x, y=y, data = df, ci=None, color=sns.xkcd_rgb['orangeish'], )
    ax2.set_ylabel(None)
    ax2.set(ylim=(0,min(ax2.get_ylim()[1]+0.05,1)))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0)) # 将纵坐标格式转化为百分比形式
    return plt

# 遍历列表所有元素
def iter_all(l, li):
    '''
    l:列表
    li:初始列表
    '''
    from collections import Iterable
    for i in l:
        if isinstance(i, Iterable):
            #当前元素为列表，继续调用
            iter_all(i, li)
        else:
            li.append(i)

# 多变量eda
def var_cross_eda(x, y, df, col, row=None, hue=None):
    '''
    x:变量X
    y:目标变量
    df:数据集
    col:按列的拆分
    row:按行的拆分
    hue:不同类别
    '''
    import warnings
    warnings.filterwarnings("ignore")
    
    # 设置双轴
    def twin_pointplot(**kwargs):
        ax = plt.twinx()
        sns.pointplot(**kwargs, ax=ax)
        ax.set_ylabel(None)
        ax.set(ylim=(0,min(ax.get_ylim()[1]+0.05,1)))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0)) # 将纵坐标格式转化为百分比形式
    # 分割图
    g = sns.FacetGrid(df, col=col, row=row, height=5, aspect=1.25) 
    # 第1个柱状图
    g.map(sns.countplot, x=x, data=df, color=sns.xkcd_rgb['windows blue'])
    # 第2个点图
    g.map(twin_pointplot, x=x, y=y, hue=hue, color=sns.xkcd_rgb['orangeish'],
          data=df, ci=None)
    g.add_legend()
    # 获取每个分割图的ax
#     axs = g.axes
#     ax_list=[]
#     iter_all(axs, ax_list)
#     # 修改每个分割图信息
#     for ax in ax_list:
#         ax.set_xlabel(x, fontsize=14)
    g.set_xlabels(x, fontsize=14)
    return plt.show()

# ---------------------------------------------------------模型指标评估---------------------------------------------------------

# 模型评估指标-混淆矩阵
def model_confusion_metrics(model, X, y, name='test'):
    '''
    model:训练的模型对象名
    X：X数据集
    y：y标签集
    name：样本名称，默认为test
    '''
    # 混淆矩阵
    pre_y = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pre_y).ravel()  # 获得混淆矩阵
    confusion_matrix_table = prettytable.PrettyTable(['','prediction-0','prediction-1'])  # 创建表格实例
    confusion_matrix_table.add_row(['actual-0',tp,fn])  # 增加第一行数据
    confusion_matrix_table.add_row(['actual-1',fp,tn])  # 增加第二行数据
    print(f'confusion matrix for {name}\n',confusion_matrix_table)
  
# 模型评估指标-核心指标  
def model_core_metrics(model, X, y, name='test'):
    '''
    model:训练的模型对象名
    X：数据集
    y：标签集
    sample：样本名称，默认为test
    '''
    # 核心评估指标
    pre_y = model.predict(X)
    y_prob = model.predict_proba(X)  # 获得决策树的预测概率，返回各标签（即0，1）的概率
    fpr, tpr, thres = roc_curve(y, y_prob[:, 1])  # ROC y_score[:, 1]取标签为1的概率，这样画出来的roc曲线为正
    auc_s = auc(fpr, tpr)  # AUC
    ks = max(tpr - fpr) # KS值
    scores = [i(y, pre_y) for  i in (accuracy_score,precision_score,\
                                         recall_score,f1_score)] # accuracy、precision、recall、f1
    scores.insert(0,auc_s)
    scores.append(ks)
    core_metrics = prettytable.PrettyTable()  # 创建表格实例
    core_metrics.field_names = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'ks']  # 定义表格列名
    core_metrics.add_row([round(i, 3) for i in scores])  # 增加数据
    print(f'core metrics for {name}\n',core_metrics)

# ---------------------------------------------------------模型指标绘图---------------------------------------------------------

# 模型区分排序能力
# ROC曲线
def plot_roc(model, X, y, name='test'):
    '''
    model:训练的模型对象名
    X：X数据集
    y：y标签集
    name：样本名称，默认为test
    
    return:返回roc曲线
    '''
    # 预处理
    y_prob = model.predict_proba(X)
    fpr, tpr, thres = roc_curve(y, y_prob[:, 1])
    # 绘图
    plt.plot(fpr, tpr, label=name)  # 通过plot()函数绘制折线图
    plt.plot([0,1],[0,1],'r--')
    plt.title('ROC Curve for') 
    plt.xlabel('FPR') 
    plt.ylabel('TPR') 
    plt.legend(loc='best') 
    return plt

# KS曲线
def plot_ks(model, X, y, name='test'):
    '''
    model:训练的模型对象名
    X：X数据集
    y：y标签集
    name：样本名称，默认为test
    
    return:返回ks曲线
    '''
    # 预处理
    y_prob = model.predict_proba(X)
    fpr, tpr, thres = roc_curve(y, y_prob[:, 1])
    # 绘图
    plt.title(f'KS Curve for {name}') 
    plt.plot(thres[1:], tpr[1:], label='tpr')
    plt.plot(thres[1:], fpr[1:], label='fpr')
    plt.plot(thres[1:], tpr[1:] - fpr[1:], label='tpr-fpr')
    plt.xlabel('threshold')
    plt.gca().invert_xaxis() 
    plt.legend(loc='best') 
    return plt

# PR曲线
def plot_pr(model, X, y, name='test'):
    '''
    model:训练的模型对象名
    X：X数据集
    y：y标签集
    name：样本名称，默认为test
    
    return:返回pr曲线
    '''
    # 预处理
    y_prob = model.predict_proba(X)
    precision, recall, _ = precision_recall_curve(y, y_prob[:, 1]) 
    # 绘图
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall,precision,label=name)
    plt.legend(loc='best') 
    return plt

# Lift曲线
def plot_lift(model, X, y, bins=10, name='test'):
    '''
    model:训练的模型对象名
    X：X数据集
    y：y标签集
    name：样本名称，默认为test
    
    return:返回lift曲线
    '''
    # 预处理
    y_prob = model.predict_proba(X)[:, 1]
    df=toad.metrics.KS_bucket(y_prob, y,
                           bucket=bins,
                           method='quantile')

    df_temp = df[['bads', 'goods', 'total']].sort_index(ascending=False).reset_index(drop=True)
    bad_total = df['bads'].sum()
    good_total = df['goods'].sum()
    all_total = bad_total + good_total
    df_temp['bad_prop']=df_temp['bads'] / bad_total
    df_temp['total_prop']=df_temp['total'] / all_total
    df_temp['cum_bads']=df_temp['bads'].cumsum()
    df_temp['cum_bad_prop']=df_temp['cum_bads'] / bad_total
    df_temp['cum_total']=df_temp['total'].cumsum()
    df_temp['cum_total_prop']=df_temp['cum_total'] / all_total
    df_temp['lift']=df_temp['bad_prop'] / df_temp['total_prop']
    df_temp['cum_lift']=df_temp['cum_bad_prop'] / df_temp['cum_total_prop']

    df_finall=df_temp[['lift', 'cum_lift']]
    random_capture=1/df_finall.shape[0]

    # 提升图
    plt.title(f'Lift Table and Cruve for {name}')
    bar_width= 0.3
    plt.bar(np.arange(df_finall.shape[0])+1,df_finall['lift'],width=bar_width,color='hotpink',label='lift')
    plt.bar(np.arange(df_finall.shape[0])+1+bar_width,random_capture,width=bar_width,color='seagreen',label='random')
    plt.xticks(np.arange(df_finall.shape[0])+1)
    plt.plot(np.arange(df_finall.shape[0])+1,df_finall['cum_lift'],'r',label='cum_lift')
    plt.xticks(np.arange(df_finall.shape[0])+1)
    plt.legend(loc='best')

# 模型泛化能力
# cv箱线图
def plot_cv_box(model, X, y, fold=5, scoring='roc_auc', name='test'):
    '''
    model:训练的模型对象名
    X：X数据集
    y：y标签集
    fold：折数，默认为5
    scoring：评价分数，默认为auc
    name：样本名称，默认为test
    
    return:返回交叉验证箱线图
    '''
    # 预处理
    cv_result = cross_val_score(estimator=model,X=X,y=y,cv=fold,n_jobs=-1,scoring=scoring)
    # 绘图
    plt.title(f'cv box for {name}')
    plt.boxplot(cv_result,patch_artist=True,showmeans=True,
    boxprops={'color':'black','facecolor':'yellow'},
    meanprops={'marker':'D','markerfacecolor':'tomato'},
    flierprops={'marker':'o','markerfacecolor':'red','color':'black'},
    medianprops={'linestyle':'--','color':'orange'})
    return plt

#学习曲线
def plot_learning_curve(model, X, y, n_splits=20, name='test'):
    '''
    model:训练的模型对象名
    X：X数据集
    y：y标签集
    n_splits：默认为20等份
    name：样本名称，默认为test
    
    return:返回学习曲线
    '''
    # 预处理
    cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    estimator = model
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # 绘图
    plt.title(f'Learning Curve for {name}')
    plt.xlabel("Training samples")
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    return plt

# 模型稳定性
def model_psi(model, X1, X2):
    '''
    model:训练的模型对象名
    X1：X1数据集
    X2：X2数据集
    
    return:模型psi
    '''
    prob1 = model.predict_proba(X1)[:, 1]
    prob2 = model.predict_proba(X2)[:, 1]
    model_psi = toad.metrics.PSI(prob1,prob2)
    return model_psi

# 模型捕获率报告
def capture_table(y_prob, y, bins=20):
    '''
    y_prob:概率
    y：真实值
    bins：等分箱数，默认为20
    
    return:模型捕获率报告
    '''
    df=toad.metrics.KS_bucket(y_prob, y,
                       bucket=bins,
                       method='quantile')
    df_temp = df[['bads', 'goods', 'bad_rate']].sort_index(ascending=False).reset_index(drop=True)
    df_temp['cum_bads']=df_temp['bads'].cumsum()
    df_temp['cum_goods']=df_temp['goods'].cumsum()
    df_temp['bads_total']=df_temp['bads'].sum()
    df_temp['goods_total']=df_temp['goods'].sum()
    df_temp['ks']=round(df_temp['cum_bads']/df_temp['bads_total'].map(lambda x:math.fabs(x))\
                           -df_temp['cum_goods']/df_temp['goods_total'],3)
    df_temp['cum_bads_prop']=df_temp['cum_bads'] / df_temp['bads_total']
    
    df_finall=df_temp[['ks', 'bads', 'goods', 'cum_bads', 'cum_goods', 'cum_bads_prop', 'bad_rate']]
    return df_finall

# 特征重要性排序
def feature_topN(model, X, n=10):
    
    # 预处理
    features = X.columns  # 获取特征名称
    importances = model.feature_importances_  # 获取特征重要性
    feature_topN = pd.Series(importances, index=features).sort_values(ascending=False)[:n]
    feature_topN.sort_values(inplace=True)
    
    # 绘图
    plt.barh(feature_topN.index, feature_topN.values, height=0.7) 
    plt.xlabel('importance') # x 轴
    plt.ylabel('features') # y轴
    plt.title('Feature Importance') # 标题
    for a,b in zip( feature_topN.values,feature_topN.index): # 添加数字标签
       plt.text(a+0.00005, b,'%.3f'%float(a)) # a+0.001代表标签位置在柱形图上方0.001处
    return plt


# ---------------------------------------------------------评分卡绘图---------------------------------------------------------

# 评分直方图
def plot_score_hist(df, y_col, score_col, cutoff=None):
    """
    df:数据集（含y_col,score列）
    y_col:目标变量的字段名
    score_col:得分的字段名
    cutoff :划分拒绝/通过的点
    
    return :好坏用户的得分分布图
    """  
    # 预处理
    x1 = df[df[y_col]==1][score_col]
    x2 = df[df[y_col]==0][score_col]
    # 绘图
    plt.title('Score_hist')
    sns.kdeplot(x1,shade=True,label='bad',color='hotpink')
    sns.kdeplot(x2,shade=True,label='good',color ='seagreen')
    if cutoff!=None:
        plt.axvline(x=cutoff)
    plt.legend()
    return plt

# 评分洛伦兹曲线
def plot_lorenz(df, y_col, score_col, bins=10, cutoff=None, name=''):
    '''
    df:数据集（含y_col,score列）
    y_col:目标变量的字段名
    score_col:得分的字段名
    bins:等分箱
    cutoff :划分拒绝/通过的点
    
    return :好坏用户的得分分布图
    '''
    score_list = list(df[score_col])
    label_list = list(df[y_col])
    items = sorted(zip(score_list,label_list),key = lambda x:x[0])
    step = round(df.shape[0]/bins,0)
    bad = df[y_col].sum()
    all_badrate = float(1/bins)
    all_badrate_list = [all_badrate]*bins
    all_badrate_cum = list(np.cumsum(all_badrate_list))
    all_badrate_cum.insert(0,0)

    score_bin_list=[]
    bad_rate_list = []
    for i in range(0,bins,1):
        index_a = int(i*step)
        index_b = int((i+1)*step)
        score = [x[0] for x in items[index_a:index_b]]
        tup1 = (min(score),)
        tup2 = (max(score),)
        score_bin = tup1+tup2
        score_bin_list.append(score_bin)
        label_bin = [x[1] for x in items[index_a:index_b]]
        bin_bad = sum(label_bin)
        bin_bad_rate = bin_bad/bad
        bad_rate_list.append(bin_bad_rate)
    bad_rate_cumsum = list(np.cumsum(bad_rate_list))
    bad_rate_cumsum.insert(0,0)
    y3 = bad_rate_cumsum
    y4 = all_badrate_cum

    plt.title('Lorenz curve')
    plt.plot(y3,color='hotpink',label='cun_bad_rate')
    plt.plot(y4,color='seagreen')
    plt.xticks(np.arange(bins+1),rotation=0)
    if cutoff!=None:
        plt.axvline(x=cutoff)
    plt.legend(loc='best')

# 寻找最优cutoff
def search_cutoff(df, y_col, score_col):
    '''
    df:数据集（含y_col,score列）
    y_col:目标变量的字段名
    score_col:得分的字段名
    
    return :最大ks及对应的score
    '''
    score_min=int(math.floor(df[score_col].min()))
    score_max=int(math.ceil(df[score_col].max()))
    
    cutoff = list(range(score_min, score_max))
    refuse_acc=[]
    tpr_ls=[]
    fpr_ls=[]
    KS_ls=[]
    for i in cutoff:
        df['result'] = df.apply(lambda x:'拒绝' if x[score_col]<=i else '接受',axis=1)
        TP = df[(df['result']=='拒绝')&(df[y_col]==1)].shape[0] 
        FN = df[(df['result']=='拒绝')&(df[y_col]==0)].shape[0] 
        bad = df[df[y_col]==1].shape[0] 
        good = df[df[y_col]==0].shape[0] 
        refuse = df[df['result']=='拒绝'].shape[0] 
        passed = df[df['result']==10].shape[0] 
        tpr = round(TP/bad,3) 
        fpr = round(FN/good,3) 
        KS = abs(tpr-fpr)
        KS_ls.append(KS)
        
    maxid_KS = KS_ls.index(max(KS_ls))
    co4 = cutoff[maxid_KS]
    print('最大KS值:{}'.format(max(KS_ls)))
    print('KS最大的分数:{}'.format(co4))                     

    return max(KS_ls),co4

# 设定cutoff并衡量有效性
def rule_verify(df, y_col, score_col, cutoff):
    """
    df:数据集
    y_col:目标变量的字段名
    score_col:得分的字段名    
    cutoff :划分拒绝/通过的点
    
    return :混淆矩阵
    """
    df['result'] = df.apply(lambda x:'拒绝' if x[score_col]<=cutoff else '接受',axis=1)
    TP = df[(df['result']=='拒绝')&(df[y_col]==1)].shape[0] 
    FN = df[(df['result']=='拒绝')&(df[y_col]==0)].shape[0] 
    bad = df[df[y_col]==1].shape[0] 
    good = df[df[y_col]==0].shape[0] 
    refuse = df[df['result']=='拒绝'].shape[0] 
    passed = df[df['result']==10].shape[0] 
     
    acc = round(TP/refuse,3) 
    tpr = round(TP/bad,3) 
    fpr = round(FN/good,3) 
    pass_rate = round(refuse/df.shape[0],3) 
    matrix_df = pd.pivot_table(df,index='result',columns=y_col,aggfunc={score_col:pd.Series.count},values=score_col) 
    
    print('拒绝准确率:{}'.format(acc))
    print('查全率:{}'.format(tpr))
    print('误伤率:{}'.format(fpr))
    print('规则拒绝率:{}'.format(pass_rate))
    return matrix_df
