import json
import pandas as pd
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt

with open('/home/stewart/workspace/results.json') as test_file:
    data = json.load(test_file)


csv_rows = []
for experiment in data:
    n_robots = experiment["Number of Robots"]
    sr_gt_ami = experiment["Single Robot GT-AMI"]
    keep_keys = ["Number of Observations", "Unique Topics", "SR-AMI", "GT-AMI"]
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
# print(df)
# sns.lineplot("Number of Observations", "SR-AMI", style="Number of Robots", hue="Method", ci=95, data=df)
# plt.show()
# sns.lineplot("Number of Observations", "GT-AMI", style="Number of Robots", hue="Method", ci=68, data=df)
# plt.show()
sns.lineplot("Number of Robots", "SR-AMI", hue="Method", ci=95, data=df)
plt.show()
sns.lineplot("Number of Robots", "GT-AMI", hue="Method", ci=95, data=df)
plt.show()