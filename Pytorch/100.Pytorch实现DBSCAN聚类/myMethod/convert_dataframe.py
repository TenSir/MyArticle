import os
import re
import pandas as pd

def getChinese(s):
    pattern="[\u4e00-\u9fa5]+"
    regex = re.compile(pattern)
    results = regex.findall(s)
    return "". join(results)


def data2dataframe(filepath):
    realfile = os.listdir(filepath)
    df = pd.DataFrame(columns=['index','category','content'])
    category_l = []
    content_l = []
    for name in realfile:
        # 获取文件中文名
        category = getChinese(name)
        filenamepath = os.path.join(filepath,name)
        doc = open(filenamepath, encoding='utf-8').read()
        category_l.append(category)
        content_l.append(doc)

    df['index'] = [i for i in range(len(category_l))]
    df['category'] = category_l
    df['content']= content_l
    df.to_csv('myDataF.csv',index = False)


data2dataframe(r'F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\THUCNews_processed')
