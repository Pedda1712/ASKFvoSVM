import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from Model import ASKFvoSVM

def lin_kernel(X1, X2, _):
        return X1 @ X2.T

def rbf_kernel(X1, X2, gamma=1):
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sqdist)

# Example Data:
use_kernel = rbf_kernel
n_classes = 3
n_dimensions = 2
n_samples = 200
X, labels = make_blobs(n_samples=n_samples,
                       centers=n_classes,
                       n_features=n_dimensions,
                       random_state=0)
                       
y_test = labels
X_test = X

# linear kernel
K1 = rbf_kernel(X, X, 0.01)
K2 = rbf_kernel(X, X, 0.1)
K3 = rbf_kernel(X, X, 1)
K4 = rbf_kernel(X, X, 10)
K5 = rbf_kernel(X, X, 100)
model = ASKFvoSVM([K1, K2, K3, K4, K5], labels)
#model = ASKFvoSVM([K3, K2, K4, K1], labels)

svs = model.getSVIndices()
K1_test = rbf_kernel(X_test, X, 0.01)
K2_test = rbf_kernel(X_test, X, 0.1)
K3_test = rbf_kernel(X_test, X, 1)
K4_test = rbf_kernel(X_test, X, 10)
K5_test = rbf_kernel(X_test, X, 100)

plabels = model.predict([K1_test, K2_test, K3_test, K4_test, K5_test])
#plabels = model.predict([K3_test, K2_test, K4_test, K1_test])
right = np.sum(np.equal(y_test, plabels))

print('got ', right, ' of ', len(y_test), ' test accuracy = ', right/len(y_test))


if n_dimensions != 2:
        plt.show()
        exit()

# visualize Separation by testing points on a grid (if model is for 2D-Data)
xs = X[:, 0]
ys = X[:, 1]
maxX = max(xs)
minX = min(xs) 
maxY = max(ys)
minY = min(ys)

res = 50

vX, vY = np.meshgrid(np.linspace(minX, maxX, res), np.linspace(minY, maxY, res))
vP = np.stack((vX.flatten(), vY.flatten()), axis=1)

Ktest1 = use_kernel(vP, X, 0.01)
Ktest2 = use_kernel(vP, X, 0.1)
Ktest3 = use_kernel(vP, X, 1)
Ktest4 = use_kernel(vP, X, 10)
Ktest5 = use_kernel(vP, X, 100)
labels = model.predict([Ktest1, Ktest2, Ktest3, Ktest4, Ktest5])
#labels = model.predict([Ktest3, Ktest2, Ktest4, Ktest1])

fig, ax = plt.subplots()
ax.scatter(vP[:,0], vP[:,1], c=labels)

fig, ax = plt.subplots()
ax.scatter(xs, ys, c=y_test)

svs = model.getSVIndices()
ax.scatter(X[svs,0], X[svs,1], c="r", marker="x")
plt.show()
