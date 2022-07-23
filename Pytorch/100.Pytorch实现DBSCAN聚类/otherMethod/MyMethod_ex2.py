import os
import jieba
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF


def plot_res(labels,trainingData,core_samples_mask):
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
    for k, col in zip(set(labels), colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        xy = trainingData[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=10)
        xy = trainingData[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
    plt.title('DBSCAN')
    plt.show()


# 1.加载数据
def load_data():
    # train、test and stop words
    traindata=pd.read_csv('data/train.csv',sep='\t')
    testdata=pd.read_csv('data/test_new.csv')

    # stopwords
    with open('data/stopwords.txt','r',encoding='utf-8') as f:
        words = f.readlines()
        stopwords = [word.strip() for word in words]
    return traindata,testdata,stopwords


# 2.文本转向量
def myword2vec(data,stopwords):
    # 这里使用精确模式，即cut_all=False模式进行分词
    data['cut_comment'] = data['comment'].apply(lambda x:' '.join(jieba.lcut(x,cut_all = False)))
    print(data['cut_comment'][:5])

    # 停用词处理
    corpus = []
    for i in range(len(data['comment'])):
        corpus.append(' '.join([word for word in jieba.lcut(data['comment'][i].strip()) if word not in stopwords]))

    # print(corpus[:3])
    # ['一如既往 好吃 希望 开 城市', '味道 不错 分量 足 客人 满意', '下雨天 想象 中 火爆 环境 干净 古色古香 做 服务行业 服务 场地 脏 阿姨 打扫']
    # return corpus

    # 使用TF-IDF
    tfidf = TFIDF(min_df=1,  # 最小支持长度
                 max_features=800000,
                 strip_accents = 'unicode',
                 analyzer = 'word',
                 token_pattern = r'\w{1,}',
                 ngram_range = (1, 2),
                 use_idf = 1,
                 smooth_idf = 1,
                 sublinear_tf = 1,
                 stop_words = None,)

    # vectorizer_word = tfidf.fit(corpus)
    # tfidf_matrix = tfidf.transform(corpus)
    # print(vectorizer_word.get_feature_names())
    # print(vectorizer_word.vocabulary_)

    # 词频矩阵
    freq_words_matrix = tfidf.fit_transform(corpus)
    # 获取词
    words = tfidf.get_feature_names()
    weight = freq_words_matrix.toarray()
    print(weight.shape)


    db = DBSCAN(eps=0.85, min_samples=5)
    result = db.fit(weight)
    source = list(db.fit_predict(weight))

    label = db.labels_

    data['res_label'] = label
    data.to_csv('1.csv')

    for i in data[data['res_label'] == 5]['comment']:
        print(i)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    plot_res(label,weight,core_samples_mask)



traindata,testdata,stopwords = load_data()
myword2vec(traindata,stopwords)

