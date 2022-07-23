# s_0 = 1
# kernal_size = [3,1,3,3,3,3,3]
# stride = [1,2,1,2,1,1,2,1] # 第一个数字固定为1，真正的s从第二个数字开始
# RF = 1
# def times(s):
#     c = 1
#     for i in s:
#         c *= i
#     return c
# for i in range(len(kernal_size)):
#     RF += (kernal_size[i]-1)*times(stride[:i+1])
#     print(RF)


