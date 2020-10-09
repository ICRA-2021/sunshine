import json
import pandas as pd
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt

# with open('/home/stewart/warp_ws/tmp2/results.json') as test_file:
with open('/home/stewart/workspace/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1600785357/10bot-results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601527836/clear-default-results/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601527836/clear-0.25-threshold-results/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601584122/seg-results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601604706/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601672384/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601906632/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601942571/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601968132/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1601968132/clear-0.25/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602044237/results.json') as test_file:
        data = json.load(test_file)


csv_rows = []
for experiment in data:
    n_robots = experiment["Number of Robots"]
    sr_gt_ami = experiment["Single Robot GT-AMI"]
    keep_keys = ["Number of Observations", "Unique Topics", "SR-AMI", "GT-AMI", "Number of Cells"]
    for method in experiment["Match Results"].keys():
        for trial in experiment["Match Results"][method]:
            row = OrderedDict()
            row["Number of Robots"] = n_robots
            row["Single Robot GT-AMI"] = sr_gt_ami
            row["Method"] = method
            for k in keep_keys:
                row[k] = trial[k]
            # print(row)
            csv_rows.append(row)
        print(csv_rows[-1])
    row = row.copy()
    row.pop("SR-AMI")
    row["Method"] = "Single Robot"
    row["GT-AMI"] = sr_gt_ami
    csv_rows.append(row)
df = pd.DataFrame(csv_rows)
df[r"Coverage (m$^2$) $\times$ GT-AMI Score"] = df["GT-AMI"] * df["Number of Cells"] / df["Number of Cells"].max()
# print(df)
ax = sns.lineplot("Number of Robots", "SR-AMI", hue="Method", ci=95, data=df[df["Number of Observations"] == df["Number of Observations"].max()])
ax.set_ylim(0, None)
plt.show()
ax = sns.lineplot("Number of Robots", "GT-AMI", hue="Method", ci=95, data=df[df["Number of Observations"] == df["Number of Observations"].max()])
ax.set_ylim(0, None)
plt.title("Merged Map Quality vs. # of Maps Merged")
plt.show()
ax = sns.lineplot("Number of Robots", r"Coverage (m$^2$) $\times$ GT-AMI Score", hue="Method", ci=95, data=df[df["Number of Observations"] == df["Number of Observations"].max()])
ax.set_ylim(0, None)
plt.title("Estimated Value of Merged Map")
plt.show()
g = sns.FacetGrid(df, col="Number of Robots", col_wrap=4)
g.map(sns.lineplot, "Number of Observations", "GT-AMI", hue="Method", ci='sd', data=df)
plt.show()
g = sns.FacetGrid(df, col="Number of Robots", col_wrap=4)
g.map(sns.lineplot, "Number of Observations", "Unique Topics", hue="Method", ci='sd', data=df)
plt.show()
# sns.lineplot("Number of Observations", "SR-AMI", style="Number of Robots", hue="Method", ci=95, data=df)
# plt.show()