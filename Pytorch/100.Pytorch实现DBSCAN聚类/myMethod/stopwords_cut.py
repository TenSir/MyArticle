import os
import re
import jieba
import pandas as pd


def getM2Chinese(s):
    # # 匹配两个字词以上的中文词语
    pattern=r'^[\u4e00-\u9fa5]{2,}$'
    regex = re.compile(pattern)
    results = regex.findall(s)
    return "". join(results)

# 创建停用词列表
def load_stopwords(stopwords_file):
    # 特殊字符列表
    stopwords = ['\u3000', '\n', ' ']
    if stopwords_file:
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.append(line.strip())
    return stopwords


def create_corpus(stopwords,rawfilepath,cut_sw_res_path):
    """
    Args:
        stopwords: 停用词列表
        rawfilepath: 原始数据文件夹路径
        cut_sw_res_path:各个文件分词保存结果根路径
    Returns:
    """
    realfile = os.listdir(rawfilepath)
    print('文本数量:',len(realfile))
    """创建实验数据语料库"""
    filename_list = []
    corpus = []
    for i in range(0, len(realfile)):
        filename = realfile[i]
        # 获取分类名称
        filename_list.append(filename[:2])
        file_full_path = os.path.join(rawfilepath, filename)  # 文档的路径
        with open(file_full_path,encoding='utf-8') as f:
            data = f.read()
            # 文本分词处理
            cut_data = jieba.cut(data,cut_all=False)
            # 过滤停用词和
            res_cut_data = ''
            for each in cut_data:
                if each not in stopwords:
                    # res_cut_data += each + ' '
                    if getM2Chinese(each):
                        res_cut_data += each + ' '
            corpus.append(res_cut_data)
            # 可以将各个不同分类文本分词、去除停用词后单独保存
            # cut_sw_res_file = os.path.join(cut_sw_res_path, filename)
            # with open(cut_sw_res_file,'w', encoding='utf-8') as resf:
                # 写入文件
                # resf.write(res_cut_data)
                # print(data_adj, file=resf)

    # 构造DataFrame数据文件
    df = pd.DataFrame(columns=['index', 'category', 'content'])
    df['index'] = [i+1 for i in range(len(corpus))]
    df['category'] = filename_list
    df['content']= corpus
    df.to_csv('myDataF_1.csv',index = False)


stopwords_file = r'F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\stopwords.txt'
rawfilepath = r'F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\data'
cut_sw_res_path = r'F:\MyArticle\Pytorch\categoryfile'

stopwords = load_stopwords(stopwords_file)
create_corpus(stopwords,rawfilepath,cut_sw_res_path)



