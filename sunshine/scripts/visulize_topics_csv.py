import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import numpy as np

data = pd.read_csv(sys.argv[1], index_col=0)
fig, ax = plt.subplots()

mat = data.to_numpy()
dist = mat / np.sum(mat, axis=1).reshape((-1, 1))

NUM_COLORS = max(10, dist.shape[1])

cm = plt.get_cmap('gist_rainbow')
color_list = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
colors = []
idx = 0
for i in range(NUM_COLORS):
    colors.append(color_list[(-1)**i * idx])
    idx += 1
ax.set_prop_cycle(color=colors)

ax.stackplot(range(dist.shape[0]), dist.transpose())
plt.show()
