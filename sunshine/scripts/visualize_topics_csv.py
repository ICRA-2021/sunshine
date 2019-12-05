import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sb
import sys
import numpy as np
import os
from tqdm import tqdm

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def plot_topics(topics_csv, colors_csv=None, repeat_start=True):
    data = pd.read_csv(topics_csv, index_col=0)
    fig, ax = plt.subplots()

    dist = data.to_numpy()
    if repeat_start:
        start = dist[0,:].reshape((1, -1))
        start = start * 1000
        start += 1
        dist = np.concatenate((start, dist), axis=0)
    dist = dist / np.sum(dist, axis=1).reshape((-1, 1))

    if colors_csv is None:
        NUM_COLORS = max(10, dist.shape[1])
        cm = plt.get_cmap('gist_rainbow')
        color_list = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
        colors = []
        idx = 0
        for i in range(NUM_COLORS):
            colors.append(color_list[(-1)**i * idx])
            idx += 1
    else:
        NUM_COLORS = dist.shape[1]
        colors = []
        with open(colors_csv, 'r') as color_list:
            for line in color_list:
                tokens = line.split(',')
                colors.append(tuple(float(v) / 255. for v in tokens[1:]))
    ax.set_prop_cycle(color=colors)
    fig.set_size_inches(6, 2)
    
    ax.get_yaxis().set_visible(False)
    plt.xlabel("Time (s)")
    loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

    ys = dist.transpose()
    ys = [ys[i, :] for i in range(ys.shape[0])]
    
    ax.stackplot(range(dist.shape[0]), *ys, colors=colors)
    return fig
    
if __name__ == "__main__":
    if os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        timeseries_files = []
        color_files = []
        for filename in os.listdir(folder):
            if filename.endswith("timeseries.csv"):
                timeseries_files.append(folder + filename)
            elif filename.endswith("colors.csv"):
                color_files.append(folder + filename)

        #timeseries_files.sort()
        #color_files.sort()
        sort_nicely(color_files)
        sort_nicely(timeseries_files)
        print(color_files[-1])
        for filename in tqdm(timeseries_files):
            assert filename.replace("timeseries", "colors") in color_files
            fig = plot_topics(filename, color_files[-1])
            plt.savefig(filename.replace("csv", "png"), dpi=240, bbox_inches='tight')
            plt.close(fig)
    elif os.path.isfile(sys.argv[1]):
        fig = plot_topics(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        plt.show()
        
