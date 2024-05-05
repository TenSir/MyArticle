import torch
from torch import nn
from torch.nn import functional as F



# from d2l import torch as d2l
# batch_size, num_steps = 32, 35
# train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')
# print(d2l.DATA_HUB['time_machine'])


# 数据加载脚本
import os
import re
import hashlib
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')

def download(name, cache_dir):
    # 判断
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                # 1048576表示读取1M的数据
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def read_time_machine(name,cache_dir):
    fname = download('time_machine',cache_dir=cache_dir)
    with open(fname, 'r') as f:
        lines = f.readlines()
    # 稍微清洗并返回列表形式的数据
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 调用
filename = 'timemachine'
cache_dir = os.path.join('.', 'data')
time_machine= read_time_machine(name=filename,cache_dir=cache_dir)
