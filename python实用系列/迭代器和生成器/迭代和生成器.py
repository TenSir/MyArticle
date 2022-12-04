'''

mylist = [1,2,3]
for i in mylist:
    print(i)

print('_______________')
# 生成器不会一次将所有的元素存入内存中，而是一边迭代一边运算：
mygenerator = (x*x for x in range(3))
for j in mygenerator:
    print(j)

print('_______________')

def createGenerator():
    for ss in range(1,4):
        list_1 =range(3)
        for m in list_1:
            yield m

mygene = createGenerator()
print(mygene)
for n in mygene:
    print(n)

print('-----------')

'''



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

list_test = [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96]
mygene = onegenerate(list_test)
print(mygene)
for n in mygene:
    print(n)

'''
list_combination =[]
list_combination_2 = []
string_index = [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96]
for number_1 in range(2, len(string_index) + 1):
    iter = itertools.combinations(string_index, number_1)
    list_combination.append(list(iter))

    # print(list(iter))
for i in range(len(list_combination)):
    # print(list_combination[i])
    for number_2 in range(len(list_combination[i])):
        each = list_combination[i][number_2]
        # print(list_combination[i][num])
        list_combination_2.append(list(each))

for each_list in list_combination_2:
    print(each_list)
'''



