import json
import numpy as np
from scipy.linalg import sqrtm
import pandas as pd
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import beta, norm
from scipy.special import logit, expit

from sklearn.mixture import GaussianMixture

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

# with open('/home/stewart/warp_ws/tmp2/results.json') as test_file:
with open('/home/stewart/workspace/results-12.json') as test_file:
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
# with open('/data/stewart/multiagent-sim-results/1602793971/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602793971-combined/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602794275/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602801949/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602806640/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1602827735/results.json') as test_file:
# with open('/data/stewart/multiagent-sim-results/1603861154/results.json') as test_file:
    data = json.load(test_file)

show_graphs = False
method_name = {
    "id": "ID Based Matching",
    "hungarian-l1": "Hungarian (TVD)",
    "hungarian-l1-dynamic": "Hungarian (TVD, Dynamic)",
    "hungarian-cos": "Hungarian (Cosine)",
    "hungarian-ato": None,#"Hungarian (ATO)",
    "clear-l1": None,#"CLEAR (TVD)",
    "clear-l1-auto": "CLEAR (TVD, Automatic Threshold)",
    "clear-l1-0.25": None,#"CLEAR (TVD, 0.25 Threshold)",
    "clear-icf-l1-0.5": "CLEAR (ICF-TVD, 0.5 Threshold)",
    "clear-l1-0.5": "CLEAR (TVD, 0.5 Threshold)",
    "clear-l1-0.75": "CLEAR (TVD, 0.75 Threshold)",
    "clear-l1-1.0": "CLEAR (TVD, 1.0 Threshold)",
    "clear-ato": "CLEAR (Adjusted TO)",
    "clear-cos": "CLEAR (Cosine)",
    "clear-cos-0.5": "CLEAR (Cosine, 0.5 Threshold)",
    "clear-cos-0.75": "CLEAR (Cosine, 0.75 Threshold)",
    "clear-cos-auto": "CLEAR (Cosine, Automatic Threshold)",
    "Single Robot": "Single Robot",
    "Single Robot Post Processed 0.5": None, #"Single Robot w/ CLEAR AT",
    "Single Robot Post Processed 0.25": None, #"Single Robot w/ CLEAR 0.25",
}

plt.rc('figure', figsize=(12.0, 10.0))
csv_rows = []
final_distances = {}
row = OrderedDict()
for experiment in data:
    n_robots = experiment["Number of Robots"]
    sr_gt_ami = experiment["Single Robot GT-AMI"]
    keep_keys = ["Unique Topics", "SR-AMI", "GT-AMI", "Number of Cells"]
    for method in experiment["Match Results"].keys():
        # print(method)
        for trial in experiment["Match Results"][method]:
            if method not in method_name:
                raise LookupError(method)
            row["Method"] = method_name[method]
            if row["Method"] is None:
                continue
            row["# of Robots"] = n_robots
            row["Mean Silhouette Index"] = sum([s for s in trial["Silhouette Indices"] if s < 1.0]) / sum([1 for s in trial["Silhouette Indices"] if s < 1.0])
            row["Mean Davies-Bouldin Index"] = sum([s for s in trial["Davies-Bouldin Indices"] if s > 0.0]) / len(trial["Davies-Bouldin Indices"])
            row["# of Observations"] = trial["Number of Observations"]
            row["Single Robot GT-AMI"] = sr_gt_ami
            for k in keep_keys:
                row[k] = trial[k]
            # print(row)
            csv_rows.append(row.copy())
        # print(csv_rows[-1])
    row = row.copy()
    row.pop("SR-AMI")
    row["Method"] = method_name["Single Robot"]
    row["GT-AMI"] = sr_gt_ami
    csv_rows.append(row.copy())
    try:
        row = row.copy()
        row["Method"] = method_name["Single Robot Post Processed 0.5"]
        srpp_gt_ami = experiment["Single Robot Post GT-AMI"]
        row["GT-AMI"] = srpp_gt_ami
        csv_rows.append(row.copy())
        row = row.copy()
        row["Method"] = method_name["Single Robot Post Processed 0.25"]
        srpp_gt_ami = experiment["Single Robot + CLEAR (0.25) GT-AMI"]
        row["GT-AMI"] = srpp_gt_ami
        csv_rows.append(row.copy())
    except:
        pass
    if n_robots == 12 and True:
        Sim_sr = np.array(experiment["Final Distances"]["clear-l1-0.75"])
        Sim_sr = Sim_sr[Sim_sr != 1]
        Sim_sr = Sim_sr[Sim_sr != 0]
        plt.figure()
        h = plt.hist(Sim_sr.flatten(), bins="fd")
        bins = h[0].size
        while Sim_sr[Sim_sr >= (bins - 1) / bins].size == 0:
            bins = bins // 2
        bin_sizes = [(Sim_sr <= i / bins).sum() - (Sim_sr <= (i - 1) / bins).sum() for i in range(1, bins + 1)]
        min_bin = np.argmin(bin_sizes)
        plt.title("Threshold = %f" % ((min_bin + 0.5) / bins))
        # xs = np.linspace(0, 1, 101)
        # ys = beta.pdf(xs, *l1_params)
        # plt.plot(xs, ys)
        plt.xticks(np.linspace(0, 1, 16))
        plt.yscale('log')
        plt.show()
    elif False:
        Sim_l1 = np.array(experiment["Final Distances"]["clear-l1-0.75"])
        Sim_l1 = Sim_l1[Sim_l1 != 1]
        Sim_l1 = Sim_l1[Sim_l1 != 0]
        Sim_l1_logit = logit(Sim_l1)
        # l1_params = beta.fit(Sim_l1)
        Sim_cos = np.array(experiment["Final Distances"]["clear-cos-0.75"])
        Sim_cos = Sim_cos[Sim_cos != 1]
        Sim_cos = Sim_cos[Sim_cos != 0]
        Sim_cos_logit = logit(Sim_cos)
        # cos_params = beta.fit(Sim_cos)
        # print("l1 max %f, cos max %f" % (Sim_l1.max(), Sim_cos.max()))
        # print(l1_params)
        # print(cos_params)
        plt.figure()
        plt.hist(Sim_l1.flatten(), bins=np.linspace(0, 1, 32))
        # xs = np.linspace(0, 1, 101)
        # ys = beta.pdf(xs, *l1_params)
        # plt.plot(xs, ys)
        plt.xticks(np.linspace(0, 1, 32))
        plt.yscale('log')
        plt.show()
        plt.figure()
        plt.hist(Sim_l1_logit.flatten(), bins=32)
        plt.show()
        # plt.figure()
        # plt.hist(Sim_cos.flatten(), bins=np.linspace(0, 1, 11))
        # ys = beta.pdf(xs, *cos_params)
        # plt.plot(xs, ys)
        # plt.xticks(np.linspace(0, 1, 11))
        # plt.yscale('log')
        # plt.show()
        # plt.figure()
        # plt.hist(Sim_cos_logit.flatten(), bins=20, density=True)
        # mixture = GaussianMixture(n_components=2).fit(Sim_cos_logit.reshape(-1, 1))
        # means_hat = mixture.means_.flatten()
        # weights_hat = mixture.weights_.flatten()
        # sds_hat = np.sqrt(mixture.covariances_).flatten()
        # xs = np.linspace(Sim_cos_logit.min(), Sim_cos_logit.max(), 100)
        # y1s = norm.pdf(xs, means_hat[0], sds_hat[0]) * weights_hat[0]
        # y2s = norm.pdf(xs, means_hat[1], sds_hat[1]) * weights_hat[1]
        # # ys = beta.pdf(xs, *cos_params)
        # plt.plot(xs, y1s)
        # plt.plot(xs, y2s)
        # plt.show()
        pass
    if n_robots == 4 and show_graphs:
        if n_robots not in final_distances:
            final_distances[n_robots] = []
        Sim = np.array(experiment["Final Distances"]["clear-l1-0.5"])
        A = Sim - np.diag(np.diag(Sim))
        # d = np.sum(A, axis=0)
        # D = np.diag(d)
        # L = D - A
        Abin = (A >= 0.5).astype(np.int)
        # Dbin = np.diag(np.sum(A, axis=0))
        # Lbin = Dbin - Abin
        # m1 = clear_estimate(D, L)
        # m2 = clear_estimate(Dbin, Lbin)
        # m3 = new_clear_estimate(d, L)

        Sim_reduced = Sim[0, :].reshape((1, -1))
        counts = []
        counter = 1
        for r in range(1, Sim.shape[0]):
            if r % int(experiment["Parameters"]["K"]) == 0:
                counts.append(counter)
                counter = 0
            if Sim[r, :r].max() < 1 and (r + 1 == Sim.shape[0] or Sim[r, r+1:].max() < 1):
                Sim_reduced = np.concatenate((Sim_reduced, Sim[r, :].reshape((1, -1))), axis=0)
                counter += 1
        if counter > 0:
            counts.append(counter)
        for c in range(Sim_reduced.shape[1] - 1, -1, -1):
            if Sim_reduced[:, c].max() == 0:
                Sim_reduced = np.concatenate((Sim_reduced[:, :c], Sim_reduced[:, c+1:]), axis=1)
        heatmap(Sim_reduced, counts)

        G = nx.from_numpy_matrix(Abin)
        nx.draw(G, pos=nx.spring_layout(G, k=0.2, iterations=50))
        plt.title("Topic Correspondences")
        plt.show()
        pass
        #
        # G = nx.from_numpy_matrix(Abin)
        # nx.draw(G)
        # plt.title("M2")
        # plt.show()
        # final_distances[n_robots].append(Sim)
df = pd.DataFrame(csv_rows)
df[r"Coverage (m$^2$) $\times$ GT-AMI Score"] = df["GT-AMI"] * df["Number of Cells"] / df["Number of Cells"].max()

# plt.rc('font', size=24)
plt.rc('figure', figsize=(20.0, 9.0 * 5/4))
sns.set(style="whitegrid", font_scale=2.25)

ax = sns.lineplot("# of Robots", "GT-AMI", hue="Method", ci=95, data=df[df["# of Observations"] == df["# of Observations"].max()])
# ax.set_ylim(0, None)
# plt.grid(True, which='both', axis='both')
plt.xlabel("Number of Merged Maps")
ax.set_xlim(1, 12)
plt.ylabel("AMI Score")
leg = ax.legend(loc="lower left")
for line in leg.get_lines():
    line.set_linewidth(6.0)
plt.title("Merged Map Quality vs. # of Maps Merged", fontsize=28)
plt.savefig("gt-ami.png")
plt.show()
ax = sns.lineplot("# of Robots", r"Coverage (m$^2$) $\times$ GT-AMI Score", hue="Method", ci=95, data=df[df["# of Observations"] == df["# of Observations"].max()])
# ax.set_ylim(0, None)
# plt.grid(True, which='both', axis='both')
plt.xlabel("Number of Merged Maps")
plt.ylabel(r"Coverage (m$^2$) $\times$ AMI Score")
ax.set_xlim(1, 12)
plt.title("Value of Merged Map vs. # of Maps Merged", fontsize=28)
leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(6.0)
plt.savefig("scaled-gt-ami.png")
plt.show()

df = df[df["Method"] != "Single Robot"]
df = df[df["Method"] != "Single Robot w/ CLEAR"]
ax = sns.lineplot("# of Robots", "SR-AMI", hue="Method", ci=95, data=df[df["# of Observations"] == df["# of Observations"].max()])
# ax.set_ylim(0, None)
# plt.grid(True, which='both', axis='both')
plt.xlabel("Number of Merged Maps")
plt.ylabel("AMI with Single Robot")
ax.set_xlim(1, 12)
plt.title("Similarity to Single Robot vs. # of Maps Merged", fontsize=28)
leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(6.0)
plt.savefig("sr-ami.png")
plt.show()

ax = sns.lineplot("# of Robots", r"Mean Silhouette Index", hue="Method", ci=95, data=df[df["# of Observations"] == df["# of Observations"].max()])
# ax.set_ylim(0, None)
# plt.grid(True, which='both', axis='both')
plt.xlabel("Number of Merged Maps")
ax.set_xlim(1, 12)
plt.ylabel("Silhouette Index (Higher is Better)")
plt.title("Silhouette Index vs. # of Maps Merged", fontsize=28)
leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(6.0)
plt.savefig("silhouette.png")
plt.show()

ax = sns.lineplot("# of Robots", r"Mean Davies-Bouldin Index", hue="Method", ci=95, data=df[df["# of Observations"] == df["# of Observations"].max()])
# ax.set_ylim(0, None)
# plt.grid(True, which='both', axis='both')
plt.ylabel("Davies-Bouldin Index (Lower is Better)")
plt.xlabel("Number of Merged Maps")
ax.set_xlim(1, 12)
plt.yscale('log')
plt.title("Davies-Bouldin Index vs. # of Maps Merged", fontsize=28)
leg = ax.legend()
for line in leg.get_lines():
    line.set_linewidth(6.0)
plt.savefig("davies-bouldin.png")
plt.show()


sns.set(style="whitegrid", font_scale=1.5)

g = sns.FacetGrid(df, col="# of Robots", col_wrap=4)
g.map(sns.lineplot, "# of Observations", "GT-AMI", hue="Method", ci='sd', data=df)
g.set(xlim=(0, None))
plt.savefig("gt-ami-obs.png")
plt.show()
g = sns.FacetGrid(df, col="# of Robots", col_wrap=4)
g.map(sns.lineplot, "# of Observations", "Unique Topics", hue="Method", ci='sd', data=df)
g.set(xlim=(0, None))
plt.savefig("topics-obs.png")
plt.show()
# sns.lineplot("Number of Observations", "SR-AMI", style="Number of Robots", hue="Method", ci=95, data=df)
# plt.show()