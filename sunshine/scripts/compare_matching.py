import pandas as pd
import csv
import ast
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def read_csv(path):
    data = []
    with open(path) as infile:
        reader = csv.DictReader(infile)
        start_time = None
        try:
            for line in reader:
                if "clear-l2" in line.values():
                    continue
                metrics = {}
                exp_keys = {}
                cluster_sizes = ast.literal_eval(line["Cluster Size"])
                if start_time is None:
                    start_time = int(line['Timestamp'])
                for key in line.keys():
                    metric = "SI" if "SI" in key else "SSD" if "SSD" in key else "DB" if "DB" in key else None
                    if metric is not None and 'Initial' not in key:
                        values = ast.literal_eval(line[key])
                        score = sum([a * b for a, b in zip(values, cluster_sizes)]) / sum(cluster_sizes)
                        dist = str(key).replace(metric, '').replace('JS', 'Jensen-Shannon').strip()
                        if dist in ['Hellinger', 'Angular', "NonMatch Probability"]:
                            continue
                        if dist in metrics:
                            metrics[dist][metric] = score
                        else:
                            metrics[dist] = {metric: score, 'Distance Metric': dist}
                    elif key == "Cluster Size":
                        values = ast.literal_eval(line[key])
                        score = sum([a * b for a, b in zip(values, cluster_sizes)]) / sum(cluster_sizes)
                        exp_keys["Average Cluster Size"] = score
                        exp_keys[key] = line[key]
                    else:
                        exp_keys[key] = line[key]
                for metric in metrics.values():
                    exp_keys.update(metric)
                    data.append(exp_keys.copy())
                end_time = int(line['Timestamp'])
        except SyntaxError:
            print("Failed to process all rows! File may be corrupt")
        except ValueError:
            print("Failed to process all rows! File may be corrupt")
    for row in data:
        row['Time'] = (float(row['Timestamp']) - start_time) / 1E9 #/ (end_time - start_time)
        # print(row['Time'])
    return data


data = []
for arg in sys.argv[1:]:
    print("Processing " + arg)
    data.extend(read_csv(arg))
df = pd.DataFrame(data)

sns.set(font_scale=1.5)
if "DB" in data[0].keys():
    g = sns.FacetGrid(df, col="Distance Metric", hue="Match Method", legend_out=True)
    # g.map(sns.regplot, "Time", "SI", x_bins=10, x_ci='sd', scatter=True, ci=None, robust=True)
    g.map(sns.regplot, "Time", "DB", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
    # g.map(sns.regplot, "Time", "SSD", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
    g.add_legend()
    g.map(plt.axhline, y=0, ls='--', c='k')
    g.set_axis_labels('Time', 'Davies-Bouldin Index')
    # sns.lineplot(x='Time', y='SI', hue='Match Method', style='Distance Metric', data=df)
    plt.tight_layout()
    plt.show()

g = sns.FacetGrid(df, col="Distance Metric", hue="Match Method", legend_out=True)
# g.map(sns.regplot, "Time", "SI", x_bins=10, x_ci='sd', scatter=True, ci=None, robust=True)
g.map(sns.regplot, "Time", "SI", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
# g.map(sns.regplot, "Time", "SSD", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
g.add_legend()
g.map(plt.axhline, y=0, ls='--', c='k')
g.set_axis_labels('Time', 'Silhouette Index')
# sns.lineplot(x='Time', y='SI', hue='Match Method', style='Distance Metric', data=df)
plt.tight_layout()
plt.show()

g = sns.FacetGrid(df, col="Distance Metric", hue="Match Method", legend_out=True)
# g.map(sns.regplot, "Time", "SI", x_bins=10, x_ci='sd', scatter=True, ci=None, robust=True)
g.map(sns.regplot, "Time", "Average Cluster Size", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
# g.map(sns.regplot, "Time", "SSD", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
g.add_legend()
# sns.lineplot(x='Time', y='SI', hue='Match Method', style='Distance Metric', data=df)
plt.tight_layout()
plt.show()

g = sns.FacetGrid(df, col="Distance Metric", hue="Match Method", legend_out=True, sharey=False)
# g.map(sns.regplot, "Time", "SI", x_bins=10, x_ci='sd', scatter=True, ci=None, robust=True)
# g.map(sns.regplot, "Time", "SI", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
g.map(sns.regplot, "Time", "SSD", x_bins=[50*i for i in range(1000//50+1)], fit_reg=False, scatter=True, ci=None, robust=True)
g.add_legend()
g.map(plt.axhline, y=0, ls='--', c='k')
g.set_axis_labels('Time', 'Mean Squared Distance to Cluster Center')
# sns.lineplot(x='Time', y='SI', hue='Match Method', style='Distance Metric', data=df)
plt.tight_layout()
plt.show()
