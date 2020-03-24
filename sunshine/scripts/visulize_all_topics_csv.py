import sys
import os
from visualize_topics_csv import plot_topics, plt
from tqdm import tqdm

folder = sys.argv[1]

timeseries_files = []
color_files = []
for filename in os.listdir(folder):
    if filename.endswith("timeseries.csv"):
        timeseries_files.append(filename)
    elif filename.endswith("colors.csv"):
        color_files.append(filename)

timeseries_files.sort()
for filename in tqdm(timeseries_files):
    try:
        if not filename.replace("timeseries", "colors") in color_files:
            raise FileNotFoundError("File %s not found" % filename.replace("timeseries", "colors"))
        fig = plot_topics(filename, filename.replace("timeseries", "colors"))
        plt.savefig(filename.replace("csv", "png"))
        plt.close(fig)
    except:
        print("Error occurred when processing file %s" % filename, file=sys.stderr)
