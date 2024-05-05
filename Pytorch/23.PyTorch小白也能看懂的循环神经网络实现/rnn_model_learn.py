
import math
import torch
# from d2l.torch import d2l
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
# time_machine= read_time_machine(name=filename,cache_dir=cache_dir)



##########################################################

import collections
def count_corpus(tokens):
    """Count token frequencies.
    Defined in :numref:`sec_text_preprocessing`"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 计算 tokens 中每个词汇的出现频率
        counter = count_corpus(tokens)
        # 对词汇及其频率进行排序，按照频率降序排列
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],reverse=True)
        # 创建一个列表，包含一个未知词汇标记 <unk> 和所有保留词汇
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 创建一个字典，将每个词汇映射到它在 idx_to_token 列表中的索引
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        # print('self.token_to_idx:', self.token_to_idx,'\n')
        # print('self.idx_to_token:',self.idx_to_token,'\n')

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    # 用于将索引转换为对应的词汇
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def load_corpus_time_machine(max_tokens=-1):
    # 加载预料
    lines = read_time_machine(name=filename,cache_dir=cache_dir)
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    # print('tokens:',tokens[:5],'\n')
    # print('corpus:', corpus[:5],'\n')
    return corpus, vocab




import torch
import random
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X),torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """两种加载数据的形式`"""
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)



def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)



# print(len(vocab))
# for X, Y in train_iter:
#     print(X,'\n')
#     print(Y,'\n')
#     print(X.T)
#     # print(X.shape)
#     # print(Y.shape)
#     break

# print(X.shape)
# torch.Size([32, 35])


# vocab_size 即词典的长度，也就是之前idx_to_token的长度
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    #隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )



def rnn(inputs, state, params):
    # inputs shape(时间步数，批量大小，词典长度)
    W_xh, W_hh, b_h, W_hq, b_q = params
    # 前一个时刻的隐藏状态，也先进行初始化
    H, = state
    outputs = []
    # X的形状：(批量大小，词典长度)
    # 之前的转置的的作用将时间步调整到最前，接着使用循环来计算每一个时间步
    for X in inputs:
        # 计算当前的H,mm(H, W_hh)中的H是上一个时间步的隐藏状态
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 使用当前的H来进行预测,Y的形状：(批量大小，词典长度)
        Y = torch.mm(H, W_hq) + b_q
        # 把结果矩阵加到列表outputs中，之后outputs的形状为 (时间步数，批量大小，词典长度)
        outputs.append(Y)
    # cat之后返回的shape为（时间步数 * 批量大小，词典长度）的矩阵
    # 隐藏层H的形状为 （批量大小，隐藏单元数）
    return torch.cat(outputs, dim=0), (H,)

##################################



class RNNModelScratch:
    '''从零开始实现的循环神经网络模型'''
    def __init__(self, vocab_size, num_hiddens, device, get_params,init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 转换为浮点数据， X.T 表示转置 X（将时间步和批次维度转置）
        # 转置之后X的shape为（时间步数 * 批量大小，词典长度）
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # print('X.shape:',X.shape) # torch.Size([35, 32, 28])
        return self.forward_fn(X, state, self.params)
    # 初始化RNN的状态
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# for X, Y in train_iter:
#     num_hiddens = 512
#     # 前向传播函数为forward_fn=rnn
#     net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params,init_rnn_state, rnn)
#     state = net.begin_state(X.shape[0], try_gpu())
#     Y, new_state = net(X.to(try_gpu()), state)
#     print('Y.shape',Y.shape)  # torch.Size([1120, 28])
#     print('len(new_state):',len(new_state))  # 1
#     print('new_state[0].shape:',new_state[0].shape) # torch.Size([32, 512])
#     break


# prefix是一段开头（用来预热）， num_preds需要生成的词数（字符）， vocab用于map成字符

def predict_ch8(prefix, num_preds, net, vocab, device):
    '''函数目的是在给定的前缀 prefix 后面生成新的字符序列'''
    state = net.begin_state(batch_size=1, device=device)
    outputs= [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # y是一个 1 x vocab 的tensor，取最大值的下标进行后续输出
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])



# num_hiddens = 512
# net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params,init_rnn_state, rnn)
# res = predict_ch8('time traveller ', 10, net, vocab, try_gpu())
# print(res)


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            # print(param)
            param.grad[:] *= theta / norm




import time
import numpy as np

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    # 进行累加
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期"""
    state, timer = None, Timer()
    metric = Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        # 如果是连续抽样的，就不用重新做初始化
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        # 转置，时间维度放在最前
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        # 计算损失
        l = loss(y_hat, y.long()).mean()
        # 如果使用的是torch.optim.Optimizer
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            # 梯度剪裁
            grad_clipping(net, 1)
            updater.step()
        # 自定义的
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


from matplotlib_inline import backend_inline
def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()



from IPython import display
from matplotlib import pyplot as plt

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,use_random_iter=False):
    """训练模型"""
    # 虽说是语言模型，但实际上是多分类问题，预测下一个字符的可能性是最大的
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity',legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 不然就调用咱自己之前搞的
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    # 输出50个字符
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        # 困惑度和每秒计算样本个数
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        # 每10个epoch看一看效果~
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_hiddens = 512
num_epochs, lr = 500, 1
net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params,init_rnn_state, rnn)
# 非随机的方式
train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu(),use_random_iter=False)