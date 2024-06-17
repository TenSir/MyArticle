import pandas as pd
data_test = pd.DataFrame({"Yes_No": [True,False],
                     "Value": [1.1,2],
                     "Type": ['Small','large']
                     })

print(data_test.dtypes)
