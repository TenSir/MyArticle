# **Pandas数据分析（四）：索引**

大家好，这里是**【Python机器学习算法实践】**，在之前关于pandas的基础教程中我们学习了Series和DataFrame的数据结构，Series和DataFrame的数据结构中有一个索引的概念，今天我们来探讨一下。

## **一、什么是Pandas索引**

索引对象的理解很简单，索引对象的作用主要是负责管理轴标签和其他元数据的。构建Series和DataFrame时，在pandas中定义的 **Index** 类来表示基本索引对象。我们来看两个打印，分别是索引对象所在类名输出，一个是其__doc__属性。

```
import pandas as pd  
print(pd.Index)  
print(pd.Index.__doc__)

输出为：

<class 'pandas.core.indexes.base.Index'>
Immutable ndarray implementing an ordered, sliceable set. The basic object
storing axis labels for all pandas objects.
Parameters
----------
data : array-like (1-dimensional)
dtype : NumPy dtype (default: object)
If dtype is None, we find the dtype that best fits the data.
If an actual dtype is provided, we coerce to that dtype if it's safe.

Otherwise, an error will be raised.
copy : bool
Make a copy of input ndarray.
name : object
Name to be stored in the index.
tupleize_cols : bool (default: True)
When True, attempt to create a MultiIndex if possible.
See Also
--------
RangeIndex : Index implementing a monotonic integer range.
CategoricalIndex : Index of :class:`Categorical` s.
MultiIndex : A multi-level, or hierarchical Index.
IntervalIndex : An Index of :class:`Interval` s.
DatetimeIndex : Index of datetime64 data.
TimedeltaIndex : Index of timedelta64 data.
PeriodIndex : Index of Period data.
Int64Index : A special case of :class:`Index` with purely integer labels.
UInt64Index : A special case of :class:`Index` with purely unsigned integer labels.
Float64Index : A special case of :class:`Index` with purely float labels.
Notes
-----
An Index instance can **only** contain hashable objects
Examples
--------
>>> pd.Index([1, 2, 3])
Int64Index([1, 2, 3], dtype='int64')
>>> pd.Index(list('abc'))
Index(['a', 'b', 'c'], dtype='object')
```

具体来说，我们查看这个Index的类的说明：

![panda_index_class](https://files.mdnice.com/user/4655/d17e9d86-1b45-40cf-b6f0-c0ad53700b65.png)panda_index_class

在官网文档的说明里，我们可以知道Index的type的种类是多样的。

**这里有一个疑问，抛给大家，大家看看能不能解决：**

上述Index(['row1', 'row2', 'row3', 'row4']数据类型明显是string类型的，为什么最后我们打印处理的dtype是object类型的呢？这是为什么呢。

大家可以参考此链接尝试自己回答上述的问题： *https://pandas.pydata.org/docs/user_guide/text.html#text-types*

其实上述截图已经很好的中的“注意点”已经说明了一切。这里大家自行学习。接下来我们看看索引如何进行创建。

## **二、为什么需要索引**

这是一个好问题，我们多这个问题进行总结： 在Pandas中，索引（Index）是一个非常重要的概念，它为数据提供了丰富的标签化和定位功能。以下是为什么Pandas中需要索引的几个关键原因：

1. **数据对齐**：Pandas中的索引确保了数据的对齐性。在进行数据操作，如算术运算或数据合并时，两个数据集的索引用于对齐数据，以便正确执行操作。
2. **快速访问**：索引提供了快速访问数据的方式。通过索引标签或整数位置，可以迅速定位到DataFrame或Series中的特定数据。
3. **分组和聚合计算**：在使用`groupby`方法进行数据分组和聚合时，索引用于指定分组的键，从而对数据进行分桶处理。
4. **排序**：索引允许对数据进行排序。可以按照索引的顺序对数据进行升序或降序排序。
5. **选择和过滤**：索引使得可以根据标签或位置选择和过滤数据变得简单。例如，可以使用布尔索引、整数索引或标签索引来选择数据子集。
6. **多级索引（MultiIndex）**：Pandas支持多级索引，即层次化索引，它允许你拥有多个维度的索引，这对于多维数据的组织和访问非常有用。
7. **性能优化**：索引可以提高数据操作的性能。Pandas内部使用索引来优化数据存取和算法，尤其是在涉及大规模数据集时。
8. **数据完整性**：索引有助于保持数据的完整性。在删除或添加数据时，索引可以确保数据的一致性和正确性。

当然了，肯定也不止上述这些原因，大家可以在评论区进行补充。

总结一下就是索引是Pandas数据处理的核心，它为数据的组织、访问、分析和操作提供了基础架构。通过有效使用索引，可以提高数据分析的效率和灵活性。所以这个索引还是非常有何必要去学习和掌握的。

## **三、索引的创建**

索引有很多的创建方法，这与Dataframe和Series的数据结构是一样的。下面举例说明一下：

**（1）基于列表创建**

基于列表来创建索引，我们可以写出如下的代码：

```
import pandas as pd
index = pd.Index([1, 2, 3, 4], name='my_index')
index
# 输出
Index([1, 2, 3, 4], dtype='int64', name='my_index')
```

这样我们就创建了一个name为“my_index”的索引，将这个索引置与DataFrame中就很好的完成了两者的合并，比如：

```
import pandas as pd
data = {'col1': [1, 2, 12, 13], 'col2': [3, 4, 34, 35 ]}
myindex = pd.Index(['row1', 'row2', 'row3', 'row4'], name='my_index')
df = pd.DataFrame(data, index=myindex)
df
```

输出 ![1](https://files.mdnice.com/user/4655/4ad70164-d7df-4fd3-8b7f-f7b76fa66c62.png)

**(2) 特定类型的索引创建**

很多时候我们需要创建不同类型的索引，比如时间序列的索引，此时Pandas也提供了很多的方法：

- DatetimeIndex：创建日期时间类型的索引

```
datetime_index = pd.DatetimeIndex(['2021-01-01', '2021-01-02'], freq='D')
datetime_index
# 输出
DatetimeIndex(['2021-01-01', '2021-01-02'], dtype='datetime64[ns]', freq='D')
```

- TimedeltaIndex：创建表示时间差的索引

```
timedelta_index = pd.TimedeltaIndex(['1 days', '2 days'])
timedelta_index
# 输出
TimedeltaIndex(['1 days', '2 days'], dtype='timedelta64[ns]', freq=None)
```

- PeriodIndex：创建表示固定时间周期的索引

```
period_index = pd.PeriodIndex(['2021-01-01', '2021-01-02'], freq='D')
period_index
# 输出
PeriodIndex(['2021-01-01', '2021-01-02'], dtype='period[D]')
```

- RangeIndex：创建一个整数范围的索引

```
range_index = pd.RangeIndex(start=0, stop=5, step=1)
range_index
# 输出
RangeIndex(start=0, stop=5, step=1)
```

我们还可以使用其他的方法来创建索引，这里就不在进行赘述了。

## **四、索引的类型**

在上面创建索引的过程中我们知道，不同的方法创建的索引的类型(type)是不一样的：

```
import pandas as pd
data = {'col1': [1, 2, 12, 13], 'col2': [3, 4, 34, 35 ]}
myindex = pd.Index(['row1', 'row2', 'row3', 'row4'], name='my_index')
df = pd.DataFrame(data, index=myindex)
print(df.index.dtype)
# 输出为：
object
```

如果我们变换一下创建的索引的类型，则：

```
import pandas as pd
data = {'col1': [1, 2, 12, 13], 'col2': [3, 4, 34, 35 ]}
myindex = pd.Index([100,101,102,103], name='my_index')
df = pd.DataFrame(data, index=myindex)
print(df.index.dtype)
# 输出
int64
```

可以看出，object和int64是不同的类型的。

从打印使用的方法可以知道，我们可以使用dtype的方法来获取index的类型，如：

![pandas index](https://files.mdnice.com/user/4655/7be2938e-aa72-434f-8dca-3f8dc0250789.png)pandas index

再次思考上述索引了类型的问题。

## **五、索引器**

这一节我们参考《joyful pandas》相关的内容。先看表的列索引。

**（1）表的列索引**

列索引是最常见的索引形式，一般通过 [] 来实现。通过 [列名] 可以从 DataFrame 中取出相应的列，返回值为Series，例如从表中取出姓名一列：

```
import pandas as pd
# 创建一个DataFrame
df = pd.DataFrame({
    'Column1': [1, 2, 3],
    'Column2': ['a', 'b', 'c'],
    'Column3': [True, False, True]
})

# 通过列索引访问列
column1_df = df['Column1']
print(type(column1_df),'\n')
column1_df
# 输出
<class 'pandas.core.series.Series'> 

0    1
1    2
2    3
Name: Column1, dtype: int64
```

如果要取出多个列，则可以通过 [列名组成的列表] ，其返回值为一个 DataFrame：

```
more_column_df = df[['Column1','Column2']]
print(type(more_column_df),'\n')
more_column_df
# 输出
<class 'pandas.core.frame.DataFrame'> 

Column1 Column2
0 1 a
1 2 b
2 3 c
```

**（2）序列的行索引**

- *以字符串为索引的 Series*

如果取出单个索引的对应元素，则可以使用 [item] ，若 Series 只有单个值对应，则返回这个标量值，如果有多个值对应，则返回一个 Series，这个还是很好理解的：

```
import pandas as pd
s = pd.Series([1, 2, 3, 4, 5, 6],
index=['a', 'b', 'a', 'a', 'a', 'c'])
print(type(s['a'],'\n'))
s['a']
# 输出
<class 'pandas.core.series.Series'>

a    1
a    3
a    4
a    5
dtype: int64
```

通过索引来访问数据：

```
print(type(s[['c', 'b']]),'\n')
s[['c', 'b']]
# 输出
<class 'pandas.core.series.Series'>

c    6
b    2
dtype: int64
```

- *以整数为索引的 Series*

这样的案例，如：

```
s = pd.Series(['a', 'b', 'c', 'd', 'e', 'f'],
               index=[1, 3, 1, 2, 5, 4])
print(type(s[1]),'\n')
s[1]
# 输出
<class 'pandas.core.series.Series'> 

1    a
1    c
dtype: object
```

另外joyful中介绍了loc和iloc的方法的使用，这里就不再赘述了，之前的文章里也有相应的介绍，如果有必要后续将对这两个方法进行专题的介绍。

## **六、索引的可重复性**

在上述的文章中我们知道，索引有存在重复的数值的，也就是说索引对象的索引是可以重复的，比如，我们在构建DataFrame的时候传入的index是可以重复的，可以这样:

```
index=['a','b','c']
```

也可以这样:

```
index=['a','b','a']
```

还可以这样:

```
index=['a','a',None]
```

我们来看一个例子：

```
import pandas as pd  
df_Test_2 = pd.DataFrame(  
{'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},  
index=['a','a','b']  
)  
print(df_Test_2)  
print(df_Test_2.index.is_unique)  
print(df_Test_2.index.isna())

# 输出为：
time count
a 2021-05-11 1
a 2021-05-12 2
b 2021-05-13 3

False

[False False False]
```

我们来解释一下上述的代码：

（1）第一行输出包含重复索引的dataframe。

（2）第二行代码使用is_unique方法来判断索引是否是唯一不重复的，返回bool值。

（3）第三行代码使用isna方法来判断索引是否是None值, index=['a','a',None]。

上述案例说明了索引的可重复性，接下来我们看看索引的不可修改性。

## **七、索引的不可修改性**

这个标题就说明索引是不支持进行修改的，但是可以进行重置。换句话说索引是不支持部分修改的。还是有点绕，我们直接来看代码演示：

```
import pandas as pd  
df_Test_3 = pd.DataFrame(  
{'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},  
index = [0,1,2]  
)  
print(df_Test_3.index[2])  
df_Test_3.index[2] = 100
# 输出
2

raise TypeError("Index does not support mutable operations")
TypeError: Index does not support mutable operations
```

之后df_Test_3.index[2] = 100报不支持修改的错： 那么该如何进行这个索引的修改的呢？毕竟在工程实践中我们有这样的要求，接下来我们介绍该如何进行重置索引：

**（1）直接赋值替换:**

```
import pandas as pd  
df_Test_3 = pd.DataFrame(  
{'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},  
index = [0,1,2]  
)df_Test_3.index=[**'a'**,**'b'**,**5**]  
print(df_Test_3)

# 输出:
time count
a 2021-05-11 1
b 2021-05-12 2
5 2021-05-13 3
```

可以发现我们将数据的索引进行“改变了”，注意看我使用的是['a','b',5]，这个列表的里面的数据的数据类型是不一样的。

**（2）使用set_index()，reset_index()和reindex()方法**

- DataFrame.set_index : Set row labels.
- DataFrame.reset_index : Remove row labels or move them to new columns.
- DataFrame.reindex : Change to new indices or expand indices._

**set_index()方法的定义如下：**

```
def set_index(  
self, keys, drop=True, append=False, inplace=False, verify_integrity=False  
)

# 参数说明
# keys：类似标签或数组的标签或标签/数组的列表  
# drop：默认为True删除要用作新索引的列  
# append：True追加到现有索引。
# inplace：是否在原数据中修改  
# verify_integrity：True检测新的索引是否重复。
import pandas as pd  
df_Test = pd.DataFrame(  
{'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},index=[1,2,3])  
  
N_df_Test = df_Test.set_index(['time'],drop=True)  
print(N_df_Test)  
```

输出为：

![img](https://files.mdnice.com/user/4655/28d2be2e-c042-48a5-8a83-bd17a3ac40e6.png)

```
new_index = N_df_Test.index.rename('这是索引哦')  
N_df_Test.index=new_index  
print(N_df_Test)
```

输出为： ![img](https://files.mdnice.com/user/4655/348fd0f2-3c50-4e99-9e78-1bb04a2c62d7.png)

上述例子大家要仔细的研究一下，另外我只举例使用**drop=True**这个参数，大家可以试一下使用其他的参数会有什么不同之处。

**reset_index()方法的定义如下：**

```
def reset_index(
level=None, drop=False, inplace=False, col_level=0, col_fill=”
)
```

- reset_index可以还原索引，重新变为默认的整型索引,和set_index()有一种相反的感觉
- level：控制了具体要还原的那个等级的索引
- drop:为False则索引列会被还原为普通列，否则会丢失

```
import pandas as pd  
df_Test = pd.DataFrame(  
{'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},  
index=list('abc'))  
print(df_Test,'\n')  
print(df_Test.reset_index,'\n')
print(df_Test.reset_index(drop=True))
```

输出为: ![img](https://files.mdnice.com/user/4655/2b560d40-200b-4291-ad58-6fcb860ef3c9.png)

大家好好体会drop=True和False的这两种情况。

这个情景非常的常见，因为在进行数据清洗的时候，我们会清洗掉一些包含NaN行的一些数据，这会造成索引不在是连贯的数值了（0,1,2…n）的数值了，这个时候我们使用上述操作就可以使得索引是连贯的。

**reset_index()方法的定义如下：**

```
DataFrame.reindex(
labels=None, index=None, columns=None, axis=None, method=None, copy=True, level=None, fill_value=nan, limit=None, tolerance=None
)
```

reindex方法可以为series和dataframe添加或者删除索引,即可以重新定义索引。

如果定义的索引没有匹配的数据，默认将已缺失值填充。

对于Series和DataFrame两个数据类型都可以通过fill_value参数填充默认值，也可以通过method参数设置填充方法。而method参数可选以下几种：

- None (默认): 不做任何填充；
- pad / ffill: 用上一行的有效数据来填充；
- backfill / bfill: 用下一行的有效数据来填充；
- nearest: 用临近行的有效数据来填充。

我们来看几个例子就明白了：

```
import pandas as pd  
df_Test_3 = pd.DataFrame(  
{'time':['2021-05-11','2021-05-12','2021-05-13'],'count':[1,2,3]},  
index = [0,1,2]  
)  
print(df_Test_3,'\n')  
print(df_Test_3.reindex([1,2,3,4,5]),'\n')  
print(df_Test_3.reindex(['a','b',5]))
```

输出为： ![img](https://files.mdnice.com/user/4655/ec0aa1a1-8531-45f0-854c-759d82f73213.png)

可以发现，新增的索引数据不在原索引对象中将引入新NaN值。

使用method参数的时候：

```
print(df_Test_3.reindex([1,2,3,4,5],method='nearest'))
# 输出
 time  count
1  2021-05-12      2
2  2021-05-13      3
3  2021-05-13      3
4  2021-05-13      3
5  2021-05-13      3
```

另外，值得注意的是reindex()可以修改列名：

```
print(df_Test_3.reindex(columns=['New_time','New_count']))
# 输出为：
New_time  New_count
0       NaN        NaN
1       NaN        NaN
2       NaN        NaN
```

由于新的列 New_count没有定义值所以填充为NaN了。

竟然还可以进行取值的操作，好比切片似的：

```
print(df_Test_3.reindex(index = [0,2],columns=['time','count']))
# 输出为：
time  count
0  2021-05-11      1
2  2021-05-13      3
```

最后我们来看看与索引相关的方法和属性：

## **八、索引对象的方法和属性**

| 方法         | 说明                                               |
| :----------- | :------------------------------------------------- |
| append       | 索引对象的连接                                     |
| delete       | 删除传入的值                                       |
| diff         | 计算差集                                           |
| intersection | 计算交集                                           |
| union        | 计算并集                                           |
| isin         | 计算一个指示各值是否都包含在参数集合中的布尔型数值 |
| unique       | 判断唯一性                                         |
| insert       | 将元素插入到指定位置                               |
| is_unique    | 检测索引值有无重复                                 |

我们举例几种的一些方法和属性来学习一下：

```
import pandas as pd  
# 创建两个索引对象  
index_1 = pd.Index(['a','a','b','c'*,'d',1,2,3,4,5,6])  
index_2 = pd.Index([1,2,3,4,5,6])  
# 创建一个series  
series_1 = pd.Series(range(11),index=index_1)  
  
# 1.判断目标索引是否存在  
print('a' in index_1)  
print(1 in index_1)  
  
# 2.长度  
print(len(index_1))  
  
# 3.切片操作,跟列表的切片是一样的  
print(index_1[:3])  
  
# 4.两个索引对象的交集  
print(index_1.intersection(index_2))  
  
# 5.两个索引对象的差集  
print(index_1.difference(index_2))  
print(index_2.difference(index_1))  
  
# 6.索引对象的连接  
print(index_1.append(index_2))  
  
# 7.删除索引(传入列表，列表中的值为删除的位置)  
print(index_1.delete([0]))  
print(index_1.delete([0,1,3]))  
  
# 8.插入  
print(index_1.insert(0,'aaa'))  
  
# 9.去重  
print(index_1.is_unique)  
print(index_1.unique())
```

输出的结果这里就不打印出来了，大家可以自己实验一下。

以上几种是比较常见Index的方法和属性举例，很好理解，这里就不解释了。大家也可以参照官网的代码的API进行学习： ![index_api](https://files.mdnice.com/user/4655/ffdab9ed-704d-4fd8-a38c-5f832593bebd.png)

## **九、总结**

本次文章我们讲解了在Pandas中最为常见的一个知识点——索引对象。当然了本次我们所见到的索引对象是一维度的，下次我们将讲解多重索引对象，所以小伙伴们先要将本次的文章的内容好好消化。MultiIndex我们下期见：

## **十、参考文档**

- https://pandas.pydata.org/docs/reference/indexing.html
- 《joyful pandas》