import pandas as pd
import numpy as np

data = pd.DataFrame({'value':[3, 67, -17, 44, 37, 3, 31, -38]})
data['log+1'] = (data['value'] + 1).transform(np.log)

#Handle Negative Values
data['log'] = (data['value'] - data['value'].min() + 1).transform(np.log)

print(data)