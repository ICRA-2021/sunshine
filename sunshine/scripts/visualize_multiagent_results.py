import json
import numpy as np
from scipy.linalg import sqrtm
import pandas as pd
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

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
# with open('/data/stewart/multiagent-sim-results/1602344955/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602344955/clear-0.5/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602344955/fixed/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602516243/seg-results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602516243/clear-0.5-new/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602558424/fixed/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602592176/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602609381/seg-results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602650022/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602743683/results.json') as test_file:
    data = json.load(test_file)

def onehot(n, x):
    z = np.zeros(n)
    z[x] = 1
    return z

def clear_estimate(D, L):
    C = D + np.eye(L.shape[0])
    C_negrt = np.real_if_close(sqrtm(np.linalg.inv(C)))
    Lnrm = np.dot(np.dot(C_negrt, L), C_negrt)
    lambdas = np.real_if_close(np.linalg.eigvals(Lnrm))
    return np.sum(lambdas <= 0.5)

def new_clear_estimate(d, L):
    S, V = np.linalg.eigh(L)
    # X = np.real(V @ np.diag(S) @ V.T)
    D = np.diag(d)
    assignments = np.argmax(np.abs(V), axis=1)
    e = S[assignments]
    e_d = e / d
    n = e_d / (e_d - 1)
    C = np.diag(n)
    C_negrt = np.real_if_close(sqrtm(np.linalg.inv(C)))
    Lnrm = np.dot(np.dot(C_negrt, L), C_negrt)
    lambdas = np.real_if_close(np.linalg.eigvals(Lnrm))
    assert np.all(n > 0)
    # assert np.all(e >= d)
    return np.sum(lambdas <= 0.5)
    # C = D + np.diag(np.mean(D, axis=0))
    # C_negrt = np.real_if_close(sqrtm(np.linalg.inv(C)))
    # Lnrm = np.dot(np.dot(C_negrt, L), C_negrt)
    # lambdas = np.real_if_close(np.linalg.eigvals(Lnrm))
    # return np.sum(lambdas <= 0.5)

def heatmap(M, counts):
    fig, ax = plt.subplots()
    im = ax.imshow(M, cmap="plasma")

    # We want to show all ticks...
    ax.set_yticks(np.arange(M.shape[0]))
    # ... and label them with the respective list entries
    ax.set_yticklabels(np.arange(1, 1 + M.shape[0]))
    # We want to show all ticks...
    ax.set_xticks(np.arange(M.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(np.arange(1, 1 + M.shape[0]))

    d_below = 0.15
    rows_needed = M.shape[1] * d_below
    y = M.shape[1] - 1 + rows_needed
    d_width = 1 / 8
    cols_needed = M.shape[0] * d_width

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.title("Topic Similarity Matrix")

    # Loop over data dimensions and create text annotations.
    # for i in range(M.shape[0]):
    #     for j in range(M.shape[1]):
    #         text = ax.text(j, i, M[i, j],
    #                        ha="center", va="center", color="w")

    plt.colorbar(im, ax=ax)
    # ax.set_title("Harvest of local farmers (in tons/year)")
    total = 0
    n = 1
    for c in counts:
        s, e = total - 0.5, total - 0.5 + c
        m = (s + e) / 2
        x = M.shape[1] - y - 1
        ax.annotate('', xy=(x, s), xytext=(x, e),  # draws an arrow from one set of coordinates to the other
                    arrowprops=dict(arrowstyle='<->', facecolor='red'),  # sets style of arrow and colour
                    annotation_clip=False)  # This enables the arrow to be outside of the plot
        ax.annotate('Robot %d' % n, xy=(x - rows_needed / 3, m - cols_needed / 2), xytext=(x - rows_needed / 3, m + cols_needed / 2),  # Adds another annotation for the text that you want
                    annotation_clip=False, rotation=90)
        ax.annotate('', xy=(s, y), xytext=(e, y),  # draws an arrow from one set of coordinates to the other
                    arrowprops=dict(arrowstyle='<->', facecolor='red'),  # sets style of arrow and colour
                    annotation_clip=False)  # This enables the arrow to be outside of the plot
        ax.annotate('Robot %d' % n, xy=(m - cols_needed / 2, y + rows_needed / 3),
                    xytext=(m - cols_needed / 2, y + rows_needed / 3),
                    # Adds another annotation for the text that you want
                    annotation_clip=False)
        total += c
        n += 1
    fig.tight_layout()
    plt.show()

csv_rows = []
final_distances = {}
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
    try:
        row = row.copy()
        row["Method"] = "Single Robot Post Processed 0.5"
        srpp_gt_ami = experiment["Single Robot Post GT-AMI"]
        row["GT-AMI"] = srpp_gt_ami
        csv_rows.append(row)
        row = row.copy()
        row["Method"] = "Single Robot Post Processed 0.25"
        srpp_gt_ami = experiment["Single Robot + CLEAR (0.25) GT-AMI"]
        row["GT-AMI"] = srpp_gt_ami
        csv_rows.append(row)
    except:
        pass
    if n_robots not in final_distances:
        final_distances[n_robots] = []
    # Sim = np.array(experiment["clear-l1"]["Final Distances"])
    # A = Sim - np.diag(np.diag(Sim))
    # d = np.sum(A, axis=0)
    # D = np.diag(d)
    # L = D - A
    # Abin = (A >= 0.5).astype(np.int)
    # Dbin = np.diag(np.sum(A, axis=0))
    # Lbin = Dbin - Abin
    # m1 = clear_estimate(D, L)
    # m2 = clear_estimate(Dbin, Lbin)
    # m3 = new_clear_estimate(d, L)
    # if n_robots == 12 and False:
    #     Sim_reduced = Sim[0, :].reshape((1, -1))
    #     counts = []
    #     counter = 1
    #     for r in range(1, Sim.shape[0]):
    #         if r % int(experiment["Parameters"]["K"]) == 0:
    #             counts.append(counter)
    #             counter = 0
    #         if Sim[r, :r].max() < 1 and (r + 1 == Sim.shape[0] or Sim[r, r+1:].max() < 1):
    #             Sim_reduced = np.concatenate((Sim_reduced, Sim[r, :].reshape((1, -1))), axis=0)
    #             counter += 1
    #     if counter > 0:
    #         counts.append(counter)
    #     for c in range(Sim_reduced.shape[1] - 1, -1, -1):
    #         if Sim_reduced[:, c].max() == 0:
    #             Sim_reduced = np.concatenate((Sim_reduced[:, :c], Sim_reduced[:, c+1:]), axis=1)
    #     heatmap(Sim_reduced, counts)
    # final_distances[n_robots].append(Sim)

    # G = nx.from_numpy_matrix(A)
    # nx.draw(G)
    # plt.title("M1")
    # plt.show()
    #
    # G = nx.from_numpy_matrix(Abin)
    # nx.draw(G)
    # plt.title("M2")
    # plt.show()
df = pd.DataFrame(csv_rows)
df[r"Coverage (m$^2$) $\times$ GT-AMI Score"] = df["GT-AMI"] * df["Number of Cells"] / df["Number of Cells"].max()
# print(df)
ax = sns.lineplot("Number of Robots", "SR-AMI", hue="Method", ci=95, data=df[df["Number of Observations"] == df["Number of Observations"].max()])
ax.set_ylim(0, None)
plt.xlabel("Number of Merged Maps")
plt.show()
ax = sns.lineplot("Number of Robots", "GT-AMI", hue="Method", ci=95, data=df[df["Number of Observations"] == df["Number of Observations"].max()])
ax.set_ylim(0, None)
plt.xlabel("Number of Merged Maps")
plt.title("Merged Map Quality vs. # of Maps Merged")
plt.show()
ax = sns.lineplot("Number of Robots", r"Coverage (m$^2$) $\times$ GT-AMI Score", hue="Method", ci=95, data=df[df["Number of Observations"] == df["Number of Observations"].max()])
ax.set_ylim(0, None)
plt.xlabel("Number of Merged Maps")
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