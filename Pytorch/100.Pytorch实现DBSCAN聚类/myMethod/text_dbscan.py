import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import manifold, metrics
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import StandardScaler


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
    return tsne_tfidf_w


def get_text_feature(filepath):
    # 读取数据
    df = pd.read_csv(filepath)
    text = df['content'].values.tolist()
    # 打乱数据
    random.shuffle(text)
    tsne_weights = calculate_tfidf(text)

    print('tsne_weights shape:',tsne_weights.shape)
    # StandardScaler一下
    tsne_weights = StandardScaler().fit_transform(tsne_weights)
    # 聚类
    # clf = DBSCAN(eps=3.5, min_samples=8)
    # clf = DBSCAN(eps=0.13, min_samples=4)
    clf = DBSCAN(eps=0.14, min_samples=8)
    y = clf.fit_predict(tsne_weights)

    # labels_属性为具体的标签
    # 画图
    # if True:
    #     fig,ax = plt.subplots()
    #     plt.scatter(tsne_weights[:, 0], tsne_weights[:, 1], c=y)
    #     plt.show()

    if True:
        fig,ax = plt.subplots()
        scatter = ax.scatter(tsne_weights[:, 0], tsne_weights[:, 1], c=y)
        legend1 = ax.legend(*scatter.legend_elements(),loc="lower left")
        ax.add_artist(legend1)
        plt.show()

    df['labels'] = pd.Series(clf.labels_)
    labels = clf.labels_
    # # labels=-1的个数除以总数，计算噪声点个数占总数的比例
    raito = df.loc[df['labels'] == -1]['content'].count() / df['content'].count()
    # 获取分簇的数目
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # # 轮廓系数评价聚类的好坏
    # my_silhouette_score = metrics.silhouette_score(tsne_weights, labels)
    # # 每一个分簇的样本数量
    every_clust_num = pd.Series(clf.labels_).value_counts()
    #
    print('分簇的数目: %d' % n_clusters_)
    print('噪声比:', format(raito, '.2%'))
    # print("轮廓系数: %0.3f" % my_silhouette_score)
    print("每一个簇下的样本数:", every_clust_num)


filepath = r"F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\myDataF.csv"
get_text_feature(filepath)










