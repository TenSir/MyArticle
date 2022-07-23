from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 预料库
corpus = ['this is the first document',
          'this document is the second document',
          'is this the first document'
          ]

# 初始化对象
vectorizer = CountVectorizer()
# 将文本中的词语装换为词频矩阵
X = vectorizer.fit_transform(corpus)
# 获取每一个词语的位置
loc = vectorizer.get_feature_names_out()
word = vectorizer.vocabulary_
# print(X)
# print(X.toarray())
# print(X.shape)
# print(loc)
# print(word)


trainsform = TfidfTransformer()
Y = trainsform.fit_transform(X)
print(Y)
print(Y.toarray())

