import itertools
def onegenerate(string_index):
    #for number_1 in range(2, len(string_index) + 1):
    for number_1 in range(len(string_index), 1, -1):
        iter = itertools.combinations(string_index, number_1)
        iter_list = list(iter)
        for number_2 in range(len(iter_list)):
            each = iter_list[number_2]
            each_list = list(each)
            yield each_list

list_test = [i for i in range(1,21)]

mygene = onegenerate(list_test)
print(mygene)
total = 0
for n in mygene:   #迭代器里面的内容
    # print(n)
    total = total + 1
print(total)

'''
class Countdown:
    def __init__(self, start):
        self.start = start

    # Forward iterator
    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1
    # Reverse iterator
    def __reversed__(self):
        n = 1
        while n <= self.start:
            yield n
            n += 1
#for rr in reversed(Countdown(30)):
#    print(rr)
for rr in Countdown(30):
    print(rr)

'''

def chunk_readtest(file, CHUNK_SIZE):
    while True:
        chunk_data = file.read(CHUNK_SIZE)
        if chunk:
            yield chunk_data
        else:
            return

f = open('yourdata')
# chunk_size为每次读入数据的块大小
CHUNK_SIZE = 2048
for chunk in chunk_readtest(f,CHUNK_SIZE):
    # 使用数据的函数
    use_chunk(chunk)

