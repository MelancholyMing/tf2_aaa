import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.io as sio
from skimage import io

# 数据可视化

mat = sio.loadmat("./data/ex7data1.mat")
print(mat.keys())

data1 = pd.DataFrame(mat.get('X'), columns=['X1', "X2"])
print(data1.head())

sns.set(context='notebook', style='white')

sns.lmplot(x='X1', y='X2', data=data1, fit_reg=False)
plt.show()

# 2-2维kmeans

mat = sio.loadmat("./data/ex7data2.mat")
data2 = pd.DataFrame(mat.get("X"), columns=["X1", "X2"])
print(data2.head())

sns.set(context='notebook', style='white')
sns.lmplot(x="X1", y="X2", data=data2, fit_reg=False)
plt.show()


def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c


def random_init(data, k):
    return data.sample(k).values


# x = np.array([1, 1])
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(x=init_centroids[:, 0], y=init_centroids[:, 1])
#
# for i, node in enumerate(init_centroids):
#     ax.annotate("{}:({},{})".format(i, node[0], node[1]), node)
#
# ax.scatter(x[0], x[1], marker='x', s=200)
# plt.show()


def _find_your_cluster(x, centroids):
    distancs = np.apply_along_axis(func1d=np.linalg.norm,
                                   axis=1,
                                   arr=centroids - x)
    return np.argmin(distancs)


# _find_your_cluster(x, init_centroids)


def assign_cluster(data, centroids):
    return np.apply_along_axis(lambda i: _find_your_cluster(i, centroids),
                               axis=1,
                               arr=data.values)


init_centroids = random_init(data2, 3)
print(init_centroids)
C = assign_cluster(data2, init_centroids)
data_with_c = combine_data_C(data2, C)
print(data_with_c.head())

sns.lmplot(x="X1", y="X2", hue='C', data=data_with_c, fit_reg=False)
plt.show()


def new_centroids(data, C):
    data_with_c = combine_data_C(data, C)
    return data_with_c.groupby("C", as_index=False).mean().sort_values(by='C').drop("C", axis=1).values


def cost(data, centroids, C):
    m = data.shape[0]
    expand_c_with_centroids = centroids[C]
    distances = np.apply_along_axis(func1d=np.linalg.norm,
                                    axis=1,
                                    arr=data.values - expand_c_with_centroids)
    return distances.sum() / m


def _k_means_iter(data, k, epoch=100, tol=0.0001):
    centroids = random_init(data, k)
    cost_progress = []

    for i in range(epoch):
        print("==============epoch {}===============".format(i))

        C = assign_cluster(data, centroids)
        centroids = new_centroids(data, C)
        cost_progress.append(cost(data, centroids, C))

        if len(cost_progress) > 1:
            if (np.abs(cost_progress[-1] - cost_progress[-2])) / cost_progress[-1] < tol:
                break
    return C, centroids, cost_progress[-1]


final_c, final_centroid, _ = _k_means_iter(data2, 3)
new_centroids(data2, C)

data_c = combine_data_C(data2, final_c)
sns.lmplot(x="X1", y="X2", hue='C', data=data_c, fit_reg=False)
plt.show()

print(cost(data2, final_centroid, final_c))


def k_means(data, k, epoch=100, n_init=10):
    tries = np.array([_k_means_iter(data, k, epoch) for _ in range(n_init)], dtype=list)
    least_cost_idx = np.argmin(tries[:, -1])

    return tries[least_cost_idx]


best_c, best_centroids, least_cost = k_means(data2, 3)
print(least_cost)

data_with_c = combine_data_C(data2, best_c)
sns.lmplot(x="X1", y="X2", hue='C', data=data_with_c, fit_reg=False)
plt.show()

from sklearn.cluster import KMeans

sk_kmeans = KMeans(n_clusters=3)

sk_kmeans.fit(data2)
sk_C = sk_kmeans.predict(data2)

data_with_c = combine_data_C(data2, sk_C)
sns.lmplot(x="X1", y="X2", hue='C', data=data_with_c, fit_reg=False)
plt.show()
