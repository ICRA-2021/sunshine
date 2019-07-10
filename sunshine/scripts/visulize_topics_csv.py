import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sys
import numpy as np

data = pd.read_csv(sys.argv[1], index_col=0, header=None)
fig, ax = plt.subplots()

mat = data.to_numpy()
dist = mat / np.sum(mat, axis=1).reshape((-1, 1))

ax.stackplot(range(dist.shape[0]), dist.transpose())
plt.show()
