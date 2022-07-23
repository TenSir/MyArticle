
import copy
import pandas as pd
import torch
import numpy as np
from sklearn import manifold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def calculate_tfidf(data):
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
    tsne = manifold.TSNE(n_components=2,random_state=0)
    tsne_tfidf_w = tsne.fit_transform(tfidf_matrix)
    print(type(tsne_tfidf_w))
    return tsne_tfidf_w


class MyDBSCAN():
    def __init__(self, eps, minpts):
        super().__init__()
        # 初始化算法参数esp和minpts
        self.eps=eps
        self.minpts=minpts

    
    def fit(self, input_data):
        self.counter=0
        self.data_len=len(input_data)
        self.data=[{'val':val, 'label':0} for val in copy.deepcopy(list(torch.tensor(input_data)))]
        self.data_matrix=copy.deepcopy(torch.tensor(input_data))
        current_label=1
        for idx, pt in enumerate(self.data):
            #print('processing: '+str(idx)+' '+str(pt))
            if pt['label']!=0:
                continue
            neighbor_loc=self.regionQuery(pt)

            if len(neighbor_loc)>=self.minpts:
                # print('Assign label '+str(current_label)+' to '+str(pt)+\
                #       ' with '+str(len(neighbor_loc))+' neighbors')
                self.data[idx]['label']=current_label
                self.counter+=1
                #print('label '+str(current_label)+': '+str(self.counter)+' points')
                self.growCluster(neighbor_loc, current_label)
            else:
                #print(str(pt)+' is noise')
                self.data[idx]['label']=-1
            self.counter=0
            current_label+=1
    
    def regionQuery(self, chosen_pt):
        distance=torch.sum(torch.sqrt((chosen_pt['val']-self.data_matrix)**2), dim=1)
        neighbor_loc=list(torch.where(distance<=self.eps)[0])
        return neighbor_loc


    def growCluster(self, neighbor_locs, current_label):
        i = 0
        while (i<len(neighbor_locs)):
            idx=neighbor_locs[i]
            if self.data[idx]['label']==-1:
                #print('Assign label '+str(current_label)+' to '+str(self.data[idx]))
                self.data[idx]['label']=current_label
                self.counter+=1
                #print('label '+str(current_label)+': '+str(self.counter)+' points')
            if self.data[idx]['label']==0:
                #print('Assign label '+str(current_label)+' to '+str(self.data[idx]))
                self.data[idx]['label']=current_label
                self.counter+=1
                #print('label '+str(current_label)+': '+str(self.counter)+' points')
                temp_neighbor_locs=self.regionQuery(self.data[idx])
                if len(temp_neighbor_locs)>=self.minpts:
                    neighbor_locs=neighbor_locs+temp_neighbor_locs
            i+=1

if __name__=="__main__":

    category = ['家居','彩票','房产','教育','股票','财经']
    lable_num = [0,1,2,3,4,5]
    df = pd.read_csv(r"F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\myDataF.csv")

    text = df['content'].values.tolist()
    tsne_weights = calculate_tfidf(text)


    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
    X = StandardScaler().fit_transform(tsne_weights)
    # print(X.shape)
    # print(type(X))
    # X = tsne_weights


    """参数寻优"""
    # eps = np.arange(0.2,4,0.1) #eps参数从0.2开始到4，每隔0.2进行一次
    # min_samples=np.arange(2,15,1)#min_samples参数从2开始到20
    # for i in eps:
    #     for j in min_samples:
    #         cls = MyDBSCAN(eps=i, minpts=j)
    #         cls.fit(torch.tensor(X))
    #         results = cls.data
    #         labels = np.array([results[i]['label'] for i in range(len(results))])
    #         # 获取分簇的数目
    #         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #         if n_clusters_ < 15 and n_clusters_ > 5:
    #             print(str(i) + "----" + str(j) + "------" + str(n_clusters_))
    #
    # exit(0)

    # cls=MyDBSCAN(eps=0.18, minpts=4)
    # 比较好
    # cls = MyDBSCAN(eps=0.17, minpts=8)  # 9类
    # cls = MyDBSCAN(eps=0.14, minpts=8)
    cls = MyDBSCAN(eps=0.17, minpts=8)
    cls.fit(torch.tensor(X))
    results=cls.data
    labels=np.array([results[i]['label'] for i in range(len(results))])
    # 获取分簇的数目
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(pd.Series(labels).value_counts())
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪点使用黑色
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()