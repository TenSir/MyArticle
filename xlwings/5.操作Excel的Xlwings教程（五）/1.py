#import xlwings as xw
#wb = xw.Book('1.xlsx')
#wb.sheets[0].pictures         # 查看引用的sheet页中图片的对象
#wb.sheets[0].pictures.count   # 统计引用的sheet页中图片对象的数量，次数输出2

#ss =  wb.sheets[0].pictures.add(r'C:\Users\LEGION\Desktop\1.jpg',width = 200,height = 200)
#print(ss)

#image:文件路径或Matplotlib图形对象。
#left :以磅为单位距离左侧边缘的位置，默认为0
#top: 以磅为单位距离上侧边缘的位置，默认为0
#width: 设置图度
#height: 设置图高
#name: Excel图片名称。如果未提供，则默认为Excel标准名称，例如“图片1”
#update: 替换更新图片
#scale: 缩放尺度

#import matplotlib.pyplot as plt
#import numpy as np
#x = [-4,-3,-2,-1,0,1,2,3,4]
#figure = plt.figure()
#plt.plot(np.cos(x)/2,np.sin(x)/3)
#wb.sheets[0].pictures.add(figure, name='sin#cos', update=True)

# picture对象是pictures集合的成员：


#wb.sheets[0].pictures[0]  # orwb.sheets[0].charts['PictureName']
#wb.sheets[0].picture[0].delete()  # 删除引用的图片
#wb.sheets[0].picture[0].height   #返回或设置图片高度。

#wb.sheets[0].picture[0].left   # 返回或设置图片水平位置。

#wb.sheets[0].picture[0].name  #返回或设置图片的名称。

#ss = wb.sheets[0].pictures[0].parent # 返回图片的父级,输出<Sheet [1.xlsx]Sheet1>
#wb.sheets[0].pictures[0].top #返回或设置图片垂直位置。

#wb.sheets[0].pictures[0].width# 返回或设置图片宽度。
#wb.sheets[0].pictures[0].update('图片路径') #用新图片替换现有图片

'''
sht = wb.sheets[0]
print(sht)
ss = sht.names
print(ss)
'''
import xlwings as xw
app = xw.App(visible=False, add_book=False)
wb = app.books.open('1.xlsx') # 打开Excel文件
sheet = wb.sheets[0]  # 选择第0个表单

sheet.names.add('python知识学堂','A1')
#保存并关闭Excel文件
wb.save()
wb.close()

