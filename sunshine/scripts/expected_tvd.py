import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import trange

beta = 0.04
V = 15436
n_samples = 500

distributions = np.random.dirichlet([beta] * V, (n_samples,))
distances = np.zeros((n_samples, n_samples))
for row in trange(n_samples):
    for col in range(row + 1, n_samples):
        distances[row][col] = np.linalg.norm(distributions[row,:] - distributions[col,:], 1)
similarity = 1 - distances/2.
data = similarity[np.triu_indices(n_samples,1)]

print(np.mean(data), np.std(data))
print(np.mean(np.log10(data)), np.std(np.log10(data)))
print(np.mean(np.log10(1 - data)), np.std(np.log10(1 - data)))

beta_model = stats.beta.fit(data)
print(beta_model)

xs = np.linspace(0, 1, num=1000)

beta_pdf = stats.beta.pdf(xs, *beta_model)

plt.figure()
plt.hist(data, bins=20, density=True)
plt.plot(xs, beta_pdf)
plt.show()

# plt.figure()
# sns.distplot(data, fit=stats.norm)
# plt.show()

plt.figure()
sns.distplot(data, fit=stats.lognorm)
plt.title('Log-Normal Fit')
plt.show()

plt.figure()
sns.distplot(data, fit=stats.beta)
plt.title('Beta Fit')
plt.show()