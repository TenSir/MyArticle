import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import manifold, metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer


# 参数寻优
def best_dbscan_param(data):
    rs= []#存放各个参数的组合计算出来的模型评估得分和噪声比
    eps = np.arange(0.2,4,0.1) #eps参数从0.2开始到4，每隔0.2进行一次
    min_samples=np.arange(2,20,1)#min_samples参数从2开始到20

    best_score=0
    best_score_eps=0
    best_score_min_samples=0
    for i in eps:
        for j in min_samples:
            try:
                db = DBSCAN(eps=i, min_samples=j).fit(data)
                labels= db.labels_
                # 轮廓系数评价聚类的好坏，值越大越好
                k=metrics.silhouette_score(data,labels)
                raito = len(labels[labels[:] == -1]) / len(labels) #计算噪声点个数占总数的比例
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
                rs.append([i,j,k,raito,n_clusters_])

                if k>best_score:
                    best_score=k
                    best_score_eps=i
                    best_score_min_samples=j
            except:
                    db='' #这里用try就是遍历i，j 计算轮廓系数会出错的，出错的就跳过
            else:
                db=''

    rs= pd.DataFrame(rs)
    rs.columns=['eps','min_samples','score','raito','n_clusters']
    sns.relplot(x="eps",y="min_samples", size='score',data=rs)
    sns.relplot(x="eps",y="min_samples", size='raito',data=rs)
    plt.show()


def plot_cluster(result, newData, numClass,clf):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
             'g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            # print ind1
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1, y1, "rv")  # 绘制中心
    plt.show()


def calculate_tfidf(corpus):
    # 分词向量化
    vectorizer = CountVectorizer()                  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    word_vec = vectorizer.fit_transform(corpus)     # 将文本转为词频矩阵
    # 提取 TF-IDF 词向量
    transformer = TfidfTransformer()                # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(word_vec)     # 计算tf-idf
    tfidf_matrix = tfidf.toarray()                  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print("TF-IDF 矩阵的维数：{}".format(tfidf_matrix.shape))
    # tsne 降维
    # tf-idf 矩阵的行数为总文档数量，列数为所有文档分词去停后所有词的数量，维数很高，而且无法在平面图上画出图形，因此需要降维
    # 降维后，可以把聚类准确率从 0.7 提高到 0.9
    tsne = manifold.TSNE(n_components=2, perplexity=20.0, early_exaggeration=12.0,
                         learning_rate=200.0, n_iter=1000, init="pca", random_state=0)
    tsne_tfidf_w = tsne.fit_transform(tfidf_matrix)
    print("降维后 TF-IDF 矩阵的维数：{}".format(tsne_tfidf_w.shape))
    return tsne_tfidf_w


def get_text_feature(filepath):
    # 读取数据
    df = pd.read_csv(filepath)
    text = df['content'].values.tolist()
    random.shuffle(text)

    for sentence in text[:5]:
        print(sentence)
    print("Vetorizier...")
    # Transfer into frequency matrix a[i][j], word j in text class i frequency
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    # vertorizer = CountVectorizer()
    # collect tf-idf weight
    transformer = TfidfTransformer()
    # outer transform for calculate tf-idf, second for transform into matrix
    tfidf = transformer.fit_transform(vertorizer.fit_transform(text))
    # get all words in BOW
    words_bag = vertorizer.get_feature_names()
    # w[i][j] represents word j's weight in text class i
    weight = tfidf.toarray()
    print('Features length:' + str(len(words_bag)))

    # 降维
    # pca = PCA(n_components=8)
    # pca_weights = pca.fit_transform(weight)
    # clf = DBSCAN(eps=0.08, min_samples=7)
    # y = clf.fit_predict(pca_weights)

    pca_weights = calculate_tfidf(text)
    clf = DBSCAN(eps=3.5, min_samples=8)
    y = clf.fit_predict(pca_weights)
    if True:
        plt.scatter(pca_weights[:, 0], pca_weights[:, 1], c=y)
        plt.show()

    # 我们看看分了所少个群，每个群的样本数是多少
    print('聚类的种类:',len(pd.Series(clf.labels_).value_counts()))
    print(pd.Series(clf.labels_).value_counts())

    # best_dbscan_param(pca_weights)

filepath = r"F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\myDataF.csv"
get_text_feature(filepath)









