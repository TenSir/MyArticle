
import copy
import pandas as pd
import torch
import numpy as np
from sklearn import manifold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class TORCHDBSCAN():
    def __init__(self, eps, minpts):
        self.eps=eps
        self.minpts=minpts

    # 计算tf-idf值
    def calculate_tfidf(self,data):
        # 分词向量化
        vectorizer = CountVectorizer()
        # 将文本转为词频矩阵
        word_v = vectorizer.fit_transform(data)
        # 提取 TF-IDF 词向量
        transformer = TfidfTransformer()
        # tf-idf
        tfidf = transformer.fit_transform(word_v)
        # 转矩阵形式
        tfidf_matrix = tfidf.toarray()
        # 对tf-idf矩阵降维为了后续的可视化
        tsne = manifold.TSNE(n_components=2, random_state=0)
        tsne_tfidf_w = tsne.fit_transform(tfidf_matrix)
        return tsne_tfidf_w


    # 此函数计算点与其他点之间的距离，然后在数据集中返回阈值距离小于eps的点的列表。
    def regionQuery(self,nowPoint,dataM):
        # 距离计算
        dist=torch.sum(torch.sqrt((nowPoint['value']-dataM)**2), dim=1)
        # 对于计算的距离小于eps的时候都加入到附近点
        neighbor_p_l = list(torch.where(dist<=self.eps)[0])
        return neighbor_p_l


    def fitPoint(self, X_matrix):
        """
            X_matrix: tensor
        """
        belong_leabel=1
        # 对每一个点进行访问
        dataM = copy.deepcopy(X_matrix)
        # 转换为字典形式如：{'value': tensor([ 0.6939, -0.1682]), 'label': 0}
        dataDict = [{'value': eachval, 'label': 0} for eachval in (list(X_matrix))]
        # 访问每一个点
        for index, each_point in enumerate(dataDict):
            # 如果这个点还没有被访问，则进行访问
            if each_point['label'] == 0:
                neighbor_p_l=self.regionQuery(each_point,dataM)
                # 如果是非噪声点(为核心点)
                if len(neighbor_p_l)>=self.minpts:
                    # 标记已经访问
                    dataDict[index]['label']=belong_leabel
                    # 开始聚类点增长
                    self.growCluster(neighbor_p_l, belong_leabel, dataM, dataDict)
                else:
                    # 噪声点标记为-1
                    dataDict[index]['label']=-1
                belong_leabel+=1
        # 返回各个点标记的数据
        return dataDict

    def growCluster(self, neighbor_p_l, belong_leabel,dataM, dataDict):
        # 停止条件
        loop = 0
        # 访问和核心点集合中的每一个点
        while (loop < len(neighbor_p_l)):
            index=neighbor_p_l[loop]
            # 非噪声数据
            if dataDict[index]['label']==0:
                dataDict[index]['label']=belong_leabel
                temp_neighbor_l=self.regionQuery(dataDict[index],dataM)
                if len(temp_neighbor_l)>=self.minpts:
                    # 合并
                    neighbor_p_l=neighbor_p_l+temp_neighbor_l
            # 噪声数据
            if dataDict[index]['label']==-1:
                dataDict[index]['label']=belong_leabel

            loop+=1


if __name__=="__main__":

    category = ['家居','彩票','房产','教育','股票','财经']
    lable_num = [0,1,2,3,4,5]
    mydf = pd.read_csv(r"F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\myDataF.csv")
    text = mydf['content'].values.tolist()

    # 初始化对象
    # cls = TORCHDBSCAN(eps=0.14, minpts=8) # (eps=0.17, minpts=8)  # 9类
    cls = TORCHDBSCAN(eps=0.17, minpts=8)  # (eps=0.17, minpts=8)
    tsne_weights = cls.calculate_tfidf(text)
    # 标准化一下我们的特征数据
    X = StandardScaler().fit_transform(tsne_weights)
    # 转换为tensor
    X_matrix = torch.tensor(X)
    # 训练
    results = cls.fitPoint(X_matrix)
    # 获取各个点的标记
    labels=np.array([results[i]['label'] for i in range(len(results))])

    # 获取分簇的数目
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 噪声点标记除以总数
    raito = list(labels).count(-1)/len(results)
    # 每一个簇别的数量
    every_clust_num = pd.Series(labels).value_counts()

    print('分簇的数目: %d' % n_clusters_)
    print('噪声比:', format(raito, '.2%'))
    print("每一个簇下的样本数:", every_clust_num)


    # 画图
    labels_unique = set(labels)
    colors = []
    for each in np.linspace(0, 1, len(labels_unique)):
        colors.append(plt.cm.Spectral(each))

    for k, col in zip(labels_unique, colors):
        if k == -1:
            # 噪点使用黑色进行标记
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

    plt.title(f'clusters num is {n_clusters_},and parameter are {"eps=0.17, minpts=8"}')
    plt.show()



