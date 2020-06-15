import numpy as np
import random
import math

from sklearn.datasets import load_iris
from sklearn.metrics.cluster import normalized_mutual_info_score


def InitCentroids(X, K):
    n = np.size(X, 0)
    rands_index = np.array(random.sample(range(1, n), K))
    centriod = X[rands_index, :]
    return centriod


def findClostestCentroids(X, centroid):
    idx = np.zeros((np.size(X, 0)), dtype=int)
    n = X.shape[0]  # n 表示样本个数
    for i in range(n):
        subs = centroid - X[i, :]
        dimension2 = np.power(subs, 2)
        dimension_s = np.sum(dimension2, axis=1)  # sum of each row
        dimension_s = np.nan_to_num(dimension_s)
        idx[i] = np.where(dimension_s == dimension_s.min())[0][0]
    return idx


def computeCentroids(X, idx, K):
    n, m = X.shape
    centriod = np.zeros((K, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]  # 一个簇一个簇的分开来计算
        temp = X[index, :]  # ? by m # 每次先取出一个簇中的所有样本
        s = np.sum(temp, axis=0)
        centriod[k, :] = s / np.size(index)
    return centriod


def costFunction(X, idx, centroids, K):
    cost = 0
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        distance2 = np.power((temp - centroids[k, :]), 2)  # ? by m
        cost = cost + np.sum(distance2)
    return cost


def isConvergence(costF, max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    for i in range(index - 1):
        if costF[i] < costF[i + 1]:
            return False
    if index >= max_iter:
        return True
    elif costF[index - 1] == costF[index - 2] == costF[index - 3]:
        return True
    return 'continue'


def kmeans(X, K, max_iter):
    costF = []
    centroids = InitCentroids(X, K)
    for i in range(max_iter):
        idx = findClostestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
        c = costFunction(X, idx, centroids, K)
        costF.append(round(c, 4))
        if i < 2:
            continue
        flag = isConvergence(costF, max_iter)
        if flag == 'continue':
            continue
        elif flag:
            best_labels = idx
            best_centers = centroids
            isConverge = True
            return isConverge, best_labels, best_centers, costF
        else:
            isConverge = False
            return isConverge, None, None, costF


class KMeans:
    def __init__(self, n_clusters=3, max_iter=20):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.isConverge, self.best_labels, self.best_centers, self.cost = kmeans(
            X=X, K=self.n_clusters, max_iter=self.max_iter
        )
        return self

    def fit_predict(self, X, y=None):
        if self.fit(X).isConverge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'

    def get_params(self):
        return self.isConverge, self.n_clusters, 'KMEAS'

    def get_cost(self):
        return self.cost


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    return x, y


if __name__ == '__main__':
    x, y = load_data()
    K = len(np.unique(y))
    model = KMeans(n_clusters=K)
    y_pred = model.fit_predict(x)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI: ", nmi)
