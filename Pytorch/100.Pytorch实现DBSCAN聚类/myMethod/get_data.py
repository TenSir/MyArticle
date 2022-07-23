import os
import shutil
import random

def get_random_text(root_path,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    category = ['财经','彩票','房产','股票','家居','教育']
    get_file_num = 300
    for i in range(len(category)):
        file_path = os.path.join(root_path, category[i])
        realfile = os.listdir(file_path)
        random_index = random.sample(range(0, len(realfile)), get_file_num)   # 随机选取

        for j in range(get_file_num):
            raw_file_path = os.path.join(root_path, category[i], realfile[random_index[j]])
            save_file_path = os.path.join(save_path, category[i] + str(j) + ".txt")
            shutil.copyfile(raw_file_path, save_file_path)
get_random_text(r'E:\DataSet\THUCNews',r'F:\MyArticle\Pytorch\Pytorch实现DBSCAN聚类\data')
