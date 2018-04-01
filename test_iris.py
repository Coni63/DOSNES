import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dosnes import dosnes

iris = datasets.load_iris()

X = iris.data
y = iris.target

metric = "sqeuclidean"

model = dosnes.DOSNES(max_iter = 1000, verbose_freq = 10, metric = metric, verbose = 1, random_state=42)
X_embedded = model.fit_transform(X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y, cmap=plt.cm.Set1)
plt.title("Iris Dataset Embedded on a Sphere with metric {}".format(metric))
plt.show()

# a = np.array([1, 2, 3])
# b = np.array(range(1, 10)).reshape(3,3)
#
# print(a)
# print("")
# print(a[np.newaxis].T)
# print("")
# print(a.reshape(-1, 1))
# print("")
# print(a + b)
# print("")
# print(a + (a.reshape(-1, 1) + b))
# print("")
# print(a + a.reshape(-1, 1) + b)