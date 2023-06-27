from openpyxl import pivot

import openpyxl.pivot.cache
import pandas as pd


pd.pivot_table(data,    # DataFrame
            values=None,    # 值
            index=None,    # 分类汇总依据
            columns=None,    # 列
            aggfunc='mean',    # 聚合函数
            fill_value=None,    # 对缺失值的填充
            margins=False,    # 是否启用总计行/列
            dropna=True,    # 删除缺失
            margins_name='All'   # 总计行/列的名称
           )


# https://www.cnblogs.com/shanger/p/13245669.html
# https://www.cnblogs.com/xiao987334176/p/14154894.html