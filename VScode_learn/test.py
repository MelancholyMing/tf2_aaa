import numpy as np
import pandas as pd
print('hello world')
d = {'a': 24, 'g': 52, 'i': 12, 'k': 33}
print(d.items())

df = pd.DataFrame([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [
                  4, 4, 4, 4]], columns=["cols1", 'cols2', 'clos3', 'clos4'])
print(df)


da = np.array([1, 2, 3, 4, 5, 6])
print(da.reshape(3, 1, 2))
