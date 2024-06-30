在 PostgreSQL 中，行转列（Row to Column）和列转行（Column to Row）是一种常见的数据转换操作，通常用于数据聚合、报告生成等场景。以下是一些基本的 SQL 语句示例，展示如何在 PostgreSQL 中进行这些转换。

### 行转列

行转列通常涉及到使用 `UNPIVOT` 语法（PostgreSQL 并不原生支持 `UNPIVOT`，但可以通过一些技巧实现类似的效果）。

假设我们有以下表格：

```
CREATE TABLE sales (
    sale_id INT,
    product VARCHAR(50),
    amount INT
);
```

并且我们想要将每个产品的销售额转换成列，可以使用以下 SQL 语句：

```
SELECT
    product,
    SUM(CASE WHEN product = 'Product1' THEN amount ELSE 0 END) AS Product1,
    SUM(CASE WHEN product = 'Product2' THEN amount ELSE 0 END) AS Product2,
    -- 以此类推
FROM
    sales
GROUP BY
    product;
```

### 列转行

列转行通常涉及到将列的值转换成行，这可以通过 `UNNEST` 函数实现，该函数用于展开数组。

假设我们有以下表格：

```
CREATE TABLE categories (
    category_id INT,
    product_ids INT[]
);
```

我们想要将 `product_ids` 数组中的每个元素转换成一行，可以使用以下 SQL 语句：

```
SELECT
    category_id,
    unnest(product_ids) AS product_id
FROM
    categories;
```

这将返回每个 `category_id` 和对应的 `product_id`。

请注意，这些示例可能需要根据你的具体需求进行调整。如果你有具体的表格结构和数据，我可以提供更具体的 SQL 语句。













import pandas as pd

### 创建一个示例 DataFrame

data = {
    'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
    'Product1': [100, 200, 300],
    'Product2': [150, 250, 350]
}
df = pd.DataFrame(data)

假设我们想要将 'Product1' 和 'Product2' 列转换为列

首先，我们需要创建一个包含所有产品名称的列表

product_list = df.columns[1:].tolist()

### 使用 pandas 的 melt 方法来实现 UNPIVOT

unpivoted_df = df.melt(id_vars=['Date'], value_vars=product_list, var_name='Product', value_name='Sales')

print(unpivoted_df)



https://zhuanlan.zhihu.com/p/159477792

https://www.jb51.net/database/303188hyz.htm



