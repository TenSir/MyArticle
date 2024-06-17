# Pandas基础教程二_Series数据结构

大家好，这里是**【Python机器学习算法与实践】**，之前在基础知识中有提及到Pandas中有两个重要的数据结构，Series和DataFrame。本期推文将先给大家讲解一些Pandas中的数据结构Series，之后再讲解DataFrame。话不说是，我们一起来学习一下。

## 一、Series简介

Series是一种一维的数据结构对象(容器)，就好比Python内置数据结构的列表。但是同的是，它显式的有一个称为索引(index)的结构，也就是说Series 是带索引的一维数组。其结构有两部分，索引和值：
![series索引和值](https://files.mdnice.com/user/4655/fc376e77-e5f3-4536-9896-76e406f4af4d.png)

**(1)索引（Index）**

- 索引是Series中每个元素的标签，可以是数字、字符串或者任何可哈希的对象。

- 索引在Series中是可选的，如果创建Series时没有指定索引，Pandas会默认创建一个从0开始的整数索引。

- 索引可以被显式地设置，这使得Series可以与数据集中的其他元素（如行名或时间戳）对齐。

- 索引是不可变的，一旦创建，就不能更改其数据类型或内容，但可以重新赋值或重新索引。

**(2)值（Values）**

- 值是Series中存储的数据，可以是任何数据类型，包括数字、字符串、布尔值或更复杂的对象。

- Series的每个值都与索引中的一个标签相关联。

- 值是可以更改的，你可以通过索引标签来访问和修改它们。

**(3)索引的重要性**
- 索引不仅用于标识数据，还可以用来进行高效的数据选择、过滤和对齐。

- 索引还可以用来进行时间序列分析，当索引是日期或时间时，Pandas可以进行时间索引和频率转换。

以上就是Series的相关的基础知识，接下来我们看看如何创  建Series.

## 二、Series的创建

在Pandas中，我们可以有很多种不同的方法可以创建Series对象。以下是一些常用的方法：

**（1）使用列表或数组：**

创建一个Series，其中列表或数组的元素成为Series的值，默认索引从0开始。
```
s = pd.Series([1, 2, 3, 4])
```
**（2）指定索引：**
```
# 为每一个元素指定索引标签
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
```

**（3）使用字典：**

使用字典创建Series，字典的键成为索引，值成为数据。
```
s = pd.Series({'a': 1, 'b': 2, 'c': 3, 'd': 4})
```

**（4）使用numpy的ndarray：**
```
import pandas as pd  
import numpy as np  
sn = pd.Series(np.random.randint(1,6),index=['A','B','C','D','E','F'])  
```
还有一种标量值的方法与上述也有类似之处：
```
s = pd.Series(5, index=['a', 'b', 'c', 'd'])
```
np.random.randint(1,6) 也就唯一产生了一个数据在1和6之间。

**（5）从函数生成：**
该方法与上述的numpy方式具有很大的相同点：
```
s = pd.Series(pd.array([1, 2, 3, 4]), index=['a', 'b', 'c', 'd'])
```
**（5）从文件生成：**
从文件中读取之后得到的DataFrame的每一列都是一个Series:
```
df = pd.read_csv('Mydata.csv')
s = df['my_column_name']
```
**（5）从时间序列生成：**
从时间序列生成的方法也是比较常见的，我们一起来看一下：
```
from pandas import date_range
s = pd.Series([1, 2, 3, 4], index=date_range('20210101', periods=4))
s

# 输出为：
2021-01-01    1
2021-01-02    2
2021-01-03    3
2021-01-04    4
Freq: D, dtype: int64
```
这种创建Series的方式还是比较优雅的，我们可以多借鉴。


## 三、Series常用属性

Series有很多的常用属性，这些属性在数据分析的时候比较常用，我们来列举几个常用的:

| 属性 | 属性说明 |
|:------|:-------|
|loc|通过索引值取数|
|iloc|通过索引位置取数|
|size|获取Series中数据量|
|shape|Series数据维度|
|dtype|Series数据类型|
|Values|获取Series的数据|

我们来看一下这些比较常见的方法：

**（1）loc**
```
import pandas as pd  
obj = pd.Series([1,2,3,4],index=['a','b','c','d'])   
  
print(obj['a'])  
print(obj['a':'c'])  
print(obj.loc['a'])  
print(obj.loc['a':'c'])  
print(obj.loc[['a','c']])

# 输出如下：  
1  
  
a 1  
b 2  
c 3  
dtype: int64  
  
1  
  
a 1  
b 2  
c 3  
dtype: int64  
  
a 1  
c 3  
dtype: int64
```
**（2）iloc**
```
import pandas as pd  
obj = pd.Series([1,2,3,4],index=['a','b','c','d'])  
print(obj.iloc[0])  
print(obj.iloc[0:2])  
print(obj.iloc[[0,2]])

##输出结果如下：
1  
  
a 1  
b 2  
dtype: int64  
  
a 1  
c 3  
dtype: int64
```

这里总结一下loc和iloc这两个属性：

**loc是对于显式索引的相关操作(对于索引)，iloc是针对隐式索引的相关操作(对于索引的位置，也就是说传入的参数是一个整数)。**

**（3）size，shape和dtypes**
```
import pandas as pd  
obj = pd.Series([1,2,3,4],index=['a','b','c','d']) 
print(obj.size)  
print(obj.shape)  
print(obj.dtype)
print(obj.values)

##输出

6

(6,)

int64

[1 2 3 4 5 6]
```
可以看出，属性values返回的由Series中数据组成的一个列表。


## 四、Series基本操作

Series中提供了很多的方法供我们使用，列举几个如下：

| 方法         | 方法说明                                 |
|--------------|-------------------------------|
| min          | 获取最小值                              |
| max          | 获取最大值                              |
| mean         | 获取平均值                              |
| median       | 获取中位数                              |
| sample       | 返回随机采样值                         |
| to_frame     | 将Series转换为DataFrame                 |
| transpose    | 进行转置                               |
| unique       | 返回唯一值组成的列表                   |
| append       | 连接两个或多个Series                   |
| corr         | 返回和另一个Series的相关系数(维度要一致) |
| cov          | 返回和另一个Series的协方差(维度要一致)   |
| describe     | 返回统计性描述                         |
| equals       | 判断两个Series是否相同(索引和值都相等)   |
| isin         | 判断元素是否在Series数据中             |
| isnull/notnull | 判断元素值是null/不是null           |
| drop_duplicates | 返回去重的Series数据               |

我们来看其中一些的一些方法，其他的方法小伙伴们自己手动代码检验：
```
import pandas as pd  
obj = pd.Series([1,2,3,4,5,6,7,7],index=['a','b','c'*,'d','e','f','g','h'])  
print(obj1.describe())  
print(obj1.sample())  
print(obj1.unique())  
print(obj1.append(obj2))  
print(obj2.equals(obj3))  
  
# 两种不同的 in 方式 
print(obj1.isin(['1','2']))  
print('a' in obj1)  
print(obj1.isnull())  
print(obj1.drop_duplicates())

#输出如下：

count 8.000000  
mean 4.375000  
std 2.263846  
min 1.000000  
25% 2.750000  
50% 4.500000  
75% 6.250000  
max 7.000000  
dtype: float64  
  
f 6  
dtype: int64  
  
[1 2 3 4 5 6 7]  
  
a 1  
b 2  
c 3  
d 4  
e 5  
f 6  
g 7  
h 7  
x 111  
y 222  
z 333  
dtype: int64  
  
False  
  
a True  
b True  
c False  
d False  
e False  
f False  
g False  
h False  
dtype: bool  
  
True  
  
a False  
b False  
c False  
d False  
e False  
f False  
g False  
h False  
dtype: bool  
  
a 1  
b 2  
c 3  
d 4  
e 5  
f 6  
g 7  
dtype: int64
```
上述方法中，我们要注意isin和in方法有什么的不同即可。

还有两个需要注意的点，这里单独提出来加强一下：

## 五、rename方法
对已存在的Series使用rename方法，可以改变其索引,方法的定义如下：（pandas 2.2版本）
![rename](https://files.mdnice.com/user/4655/a01f5f91-e78a-446b-9c27-b35cd8db891d.png)

我们来看一下是怎么使用的：
```
s = pd.Series([1, 2, 3])
print(s.name)
print(s)
# 输出
None
0    1
1    2
2    3
dtype: int64
```
改变Series的名字：
```
# 改变Series的名字
s.rename("my_name",inplace=True)
print(s.name)
print(s)
# 输出
my_name
0    1
1    2
2    3
Name: my_name, dtype: int64
```
使用lambda改变标签:
```
s.rename(lambda x: x ** 2,inplace = True)
print(s.name)
print(s)
# 输出
my_name
0    1
1    2
4    3
Name: my_name, dtype: int64
```
使用字典映射改变标签:
```
s.rename({1: 3, 2: 5},inplace = True)
print(s.name)
print(s)
# 输出
my_name
0    1
3    2
4    3
Name: my_name, dtype: int64
```
通过lable标签来访问数据：
```
values = s[[0,3]]
values
# 输出
0    1
3    2
Name: my_name, dtype: int64

# 以下报错,KeyError: '[1] not in index'
values = s[[0,1]]
values
```

## 六、reindex方法
reindex()方法也是非常常见的，我们来简单的看一下这个函数的定义：
![reindex](https://files.mdnice.com/user/4655/eb39e048-75ba-42b3-9459-7626a65f6e56.png)

这个函数为什么那么常见呢，因为：Pandas中的Series对象的reindex方法允许你重新对Series进行索引，即改变Series的索引。这个方法非常有用，特别是当你需要将Series与另一个具有不同索引的DataFrame或Series对齐时：
```
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
new_s = s.reindex(['a', 'b', 'd'])
new_s  
# 输出
a    1.0
b    2.0
d    NaN
dtype: float64
```
对已存在的Series使用reindex方法，可以改变其索引，但可能会引入NaN值。

再如：
```
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
new_s = s.reindex(['a', 'b', 'd'], method='pad')
new_s
# 输出
a    1
b    2
d    3
dtype: int64
```
所以这里的method参数就起到了关键性的作用，这个参数解释如下：
![mthod的parm](https://files.mdnice.com/user/4655/f62b311d-39e1-481b-9490-aa65ca62c718.png)
我就不翻译了，大家自己看。

当然了还有一个**limit**参数来限制其中替换测次数：
```
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
new_s = s.reindex(['a', 'b', 'c', 'd', 'e'], method='pad', limit=1)
new_s
# 输出
a    1.0
b    2.0
c    3.0
d    3.0
e    NaN
dtype: float64
```
reindex函数也是很重要的。
**很多时候我们在进行数据分析的时候。从一个Series中抽出了一部分的数据而没有进行reindex的操作。会造成很多的Bug,这点大家多注意，我也是踩坑过来的。**


## 七、Series其他操作

另外在Series中，还有其他的操作，我们来看几个比较常用的操作，基于bool数组对Series数据进行过滤和数学函数的使用：
```
import pandas as pd  
obj1 = pd.Series([1,2,3,4,5,6,7,7],
      index=['a','b','c','d','e','f','g','h'])  
print(obj1[obj1 > 5])

# 输出：
f 6
g 7
h 7
dtype: int64
```
还可以这样：
```
import pandas as pd  
obj1 = pd.Series([1,2,3,4,5,6,7,7],
      index=['a','b','c','d','e','f','g','h'])  
print(obj1[obj1 > obj1.mean()])
```
基于数学函数的使用如：
```
import pandas as pd  
import numpy as np  
obj1 = pd.Series([1,2,3,4,5,6,7,7],
      index=['a','b','c','d','e','f','g','h']) 
print(np.exp(obj1)/2)
# 输出：
a 1.359141
b 3.694528
c 10.042768
d 27.299075
e 74.206580
f 201.714397
g 548.316579
h 548.316579
dtype: float64
```
在官方文档中Series中还有很多其他的方法大家可以学习：
**https://pandas.pydata.org/docs/reference/api/pandas.Series.html**
![series其他属性和方法](https://files.mdnice.com/user/4655/1f37c3c4-e2c8-4150-bb47-f4d54cd2bf91.png)

## 八、总结
Series是Pandas中重要的数据结构，很多时候我们基于数据的操作就是在操作Series，因此掌握好Pandas必须要掌握好Series的用法和一些常用的函数。

我是一名金融科技从业者。感谢你花时间阅读我的文章，我希望我们能成为朋友。


**不为盛名而来，不为低谷而去。**

**文章就到这里，既然看到这里了，如果觉得不错，随手点个**赞**和在看吧。**


[个人知乎](https://www.zhihu.com/people/10xiansheng)
感谢关注





