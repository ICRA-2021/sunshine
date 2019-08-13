import os
from visualize_topics_csv import plot_topics

folder = sys.argv[1]

timeseries_files = []
color_files = []
for filename in os.listdir(directory):
    if filename.endswith("timeseries.csv"):
        timeseries_files.append(filename)
    elif filename.endswith("colors.csv"):
        color_files.append(filename)

timeseries_files.sort()
for filename in timeseries_files:
    assert filename.replace("timeseries", "colors") in color_files
    fig = plot_topics(filename, filename.replace("timeseries", "colors"))
    plt.savefig(filename.replace("csv", "png"))
