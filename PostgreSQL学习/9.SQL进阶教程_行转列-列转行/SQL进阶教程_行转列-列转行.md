 

# SQL进阶教程_行转列-列转行



在数据开发的过程，我们经常需要进行 行转列（Row to Column）和列转行（Column to Row）是一种常见的数据转换操作，有时候在数据库中有一个字段是多个值拼接在一起。以下是一些基本的 SQL 语句示例，展示如何在 PostgreSQL 中进行这些转换。

### 一、行转列

行转列通常涉及到使用 `UNPIVOT` 语法（PostgreSQL 并不原生支持 `UNPIVOT`，但可以通过一些技巧实现类似的效果）。假设我们有以下数据表test_hl：

```sql

CREATE TABLE test_hl (
    id VARCHAR(50),
    subject VARCHAR(50),
    score INT
);

INSERT INTO test_hl VALUES ('小红', '语文', 96);
INSERT INTO test_hl VALUES ('小红', '数学', 97);
INSERT INTO test_hl VALUES ('小红', '外语', 98);
INSERT INTO test_hl VALUES ('小明', '语文', 94);
INSERT INTO test_hl VALUES ('小明', '数学', 85);
INSERT INTO test_hl VALUES ('小明', '外语', 96);
INSERT INTO test_hl VALUES ('小蓝', '数学', 97);
INSERT INTO test_hl VALUES ('小蓝', '外语', 98);
INSERT INTO test_hl VALUES ('小蓝', '语文', 98)
```

![](1.png)

## 1.1 case-when方法

如果我们想将每个同学的成绩转换成列，可以使用以下 SQL 语句：

```sql
select id,
	sum(case when subject='语文' then score else 0 end) as 语文,
	sum(case when subject='数学' then score else 0 end) as 数学,
	sum(case when subject='外语' then score else 0 end) as 外语
from test_hl 
group by id
```

输出如下：

![](2.png)



大家注意上述sum中使用的case-when方法和之后的else=0的用法即可。



## 1.2 使用crosstab交叉函数

这个函数还是很有用的，我们看一下基于Postgresql的crosstab交叉函数该怎么进行行传列的操作呢？我们一起看一下：

```sql
select * from
crosstab
(
	'select id,
	subject,
	score from 
	test_hl ',
$$values('语文'), ('数学'), ('外语')$$
)
as score(name text, 语文 int, 数学 int, 外语 int)
```

输出的结果与上述一样，这里就不在赘述了。通过官方文档我们知道crosstab的说明如下：

![](3.png)

官网介绍到，该函数输出的是一种类似于Excel中“透视表”的样式的数据。我们在看一种方法怎么来生成行转列的方式。



## 1.3 split_part和string_agg

这个方法还是比较巧妙的，先使用string_agg方法将数据拼接到一起，然后再进行拆解，我们先看string_agg方法：

```sql
select id, string_agg(subject||':'||score,',') as temp from test_hl
group by id 
```

通过string_agg方法将数据进行拼接，输出如下：

![](4.png)



然后我们基于上述的temp表进行数据的进行第一次拆分：

```sql
select id, 
split_part(temp,',',1) as 语文,
split_part(temp,',',2) as 数学,
split_part(temp,',',3) as 外语

from
(
	select id, string_agg(subject||':'||score,',') as temp from test_hl
	group by id 
) as t
```

得到结果如下：

![](5.png)



之后再进行一次数据的拆分，就可以得到最终的结果，拆分如下：

```sql
select id, 
split_part(split_part(temp,',',1),':',2) as 语文,
split_part(split_part(temp,',',2),':',2) as 数学,
split_part(split_part(temp,',',3),':',2) as 外语

from
(
	select id, string_agg(subject||':'||score,',') as temp from test_hl
	group by id 
) as t
```

输出如下：

![](6.png)



上述一些方法就可以帮助我们解决行转列的问题，接下来我们看一下怎么进行列转行的操作。





# 二、列转行

列转行通常涉及到将列的值转换成行，假设我们有一个表，表中的数据如下：
```sql
CREATE TABLE test_lh (
    id VARCHAR(50),
    sub_score VARCHAR(50)
);

INSERT INTO test_lh VALUES ('小蓝', '数学:97,外语:98,语文:98');
INSERT INTO test_lh VALUES ('小红', '语文:96,数学:97,外语:98');
INSERT INTO test_lh VALUES ('小明', '语文:94,数学:85,外语:96')
```

数据展示如下：

![](7.png)



## 2.1 unset方法

我们该数据进行拆分，这里假设就拆分为一次，我们这里使用unnest函数，该函数的使用方法如下：

```sql
unnest(string_to_array(需要拆分的字段, '拆分符号')) as 你定义的字段新名称
```

我们可以写出以下的代码：

```sql
select 
	id 
	,sub_score
	, unnest(string_to_array(sub_score, ',')) as split_sub_score 
from test_lh 
```

这里也保留sub_score字段，输出结果如下：

![](8.png)

这样我们就将 sub_score中中的每个元素转换成一行。还算是比较方便的。



## 2.2 regexp_split_to_table方法

在PostgreSQL中我们还可以使用更加方便的函数-regexp_split_to_table,该方法的使用如下：

![](9.png)

regexp_split_to_table函数可以将字符串根据正则表达式分割成多个行。这对于分割以逗号分隔的字符串特别有用.

因此对于上述的例子。我们可以这样：

```sql
SELECT 
  id,
  regexp_split_to_table(sub_score, ',') AS split_sub_score 
FROM 
  test_lh;
```

输出：

![](10.png)





## 3.python实现行转列和列转行

## 3.1 行转列

使用pandas来实现就比较容易了，参考前文，我们使用透视表**pivot_table**来实现：

```python
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
# 连接数据库取数
engine = create_engine('postgresql+psycopg2://postgres:123456@127.0.0.1:5432/sql_advanced') 
df = pd.read_sql_query('SELECT * FROM test_hl', engine) 


# 使用 pivot_table 函数将行数据转换为列数据
result = df.pivot_table(index='id', columns='subject', values='score')

# 如果你想要与 PostgreSQL 中的结果一样，可以重命名列名
result.columns = [f'{col[1]}_{col[0]}' for col in result.columns]
result
```

上述代码输出为：

![](11.png)



如果输出结果的index,则有：

![](12.png)

可以看出索引的名称为name = 'id',为了保持数据好看，我们这里将在这个name移除，代码可以改写为：

```python
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
# 连接数据库取数
engine = create_engine('postgresql+psycopg2://postgres:123456@127.0.0.1:5432/sql_advanced') 
df = pd.read_sql_query('SELECT * FROM test_hl', engine) 


# 使用 pivot_table 函数将行数据转换为列数据
result = df.pivot_table(index='id', columns='subject', values='score')

# 如果你想要与 PostgreSQL 中的结果一样，可以重命名列名
result.columns = [f'{col[1]}_{col[0]}' for col in result.columns]

# 删除索引的名称
result.index.name = None

result
```

这样我们的输出则为：

![](13.png)



## 3.2 列转行

使用pandas来实现列转行，我们先创建一个数据：

```python
import pandas as pd

# 创建一个示例 DataFrame

data = {
    'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
    'Product1': [100, 200, 300],
    'Product2': [150, 250, 350]
}
df = pd.DataFrame(data)
df
```

![](14.png)



假设我们想要将 'Product1' 和 'Product2' 列转换为列首先，我们需要创建一个包含所有产品名称的列表,接着使用 pandas 的 melt 方法来实现：

```python
product_list = df.columns[1:].tolist()
unpivoted_df = df.melt(id_vars=['Date'], value_vars=product_list, var_name='Product', value_name='Sales')
unpivoted_df
```

![](15.png)

# 4.参考文献

https://www.postgresql.org/docs/14



