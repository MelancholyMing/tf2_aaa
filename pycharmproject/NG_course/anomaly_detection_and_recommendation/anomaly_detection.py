# 异常检测

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='notebook', style="white", palette=sns.color_palette("RdBu"))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

mat = sio.loadmat("./data/ex8data1.mat")
print(mat.keys())

X = mat.get("X")

Xval, Xtest, yval, ytest = train_test_split(mat.get("Xval"), mat.get("yval").ravel(), test_size=0.5)
sns.regplot(x="Latency", y="Throughput", data=pd.DataFrame(X, columns=['Latency', "Throughput"]),
            fit_reg=False,
            scatter_kws={'s': 20, 'alpha': 0.5})
plt.show()

mu = X.mean(axis=0)
print("mens:", mu, '\n')

cov = np.cov(X.T)
print("cov:", cov)

# np.dstack(np.mgrid[0:3, 0:3])

multi_normal = stats.multivariate_normal(mu, cov)

x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))

fig, ax = plt.subplots()

ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')
plt.show()


def select_threshold(X, Xval, yval):
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    pval = multi_normal.pdf(Xval)
    epsilon = np.linspace(np.min(pval), np.max(pval), num=10000)
    fs = []
    for e in epsilon:
        y_pred = (pval <= e).astype('int')
        fs.append(f1_score(yval, y_pred))

    argmax_fs = np.argmax(fs)
    return epsilon[argmax_fs], fs[argmax_fs]


e, fs = select_threshold(X, Xval, yval)
print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))


def predict(X, Xval, e, Xtest, ytest):
    Xdata = np.concatenate((X, Xval), axis=0)
    mu = Xdata.mean(axis=0)
    cov = np.cov(Xdata.T)
    multi_normal = stats.multivariate_normal(mu, cov)

    pval = multi_normal.pdf(Xtest)
    y_pred = (pval <= e).astype('int')

    print(classification_report(ytest, y_pred))

    return multi_normal, y_pred


multi_normal, y_pred = predict(X, Xval, e, Xtest, ytest)

data = pd.DataFrame(Xtest, columns=['Latency', 'Throughput'])
data['y_pred'] = y_pred

x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))

fig, ax = plt.subplots()

ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

sns.regplot(x="Latency", y='Throughput',
            data=data,
            fit_reg=False,
            ax=ax,
            scatter_kws={'s': 10, 'alpha': 0.4})

anomaly_data = data[data['y_pred'] == 1]
ax.scatter(anomaly_data['Latency'], anomaly_data['Throughput'],
           marker='x', s=50)
plt.show()

mat = sio.loadmat("./data/ex8data2.mat")
X = mat.get("X")
Xval, Xtest, yval, ytest = train_test_split(mat.get("Xval"),
                                            mat.get("yval").ravel(),
                                            test_size=0.5)

e, fs = select_threshold(X, Xval, yval)
print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))

multi_normal, y_pred = predict(X, Xval, e, Xtest, ytest)

print("find {} anomlies".format(y_pred.sum()))

