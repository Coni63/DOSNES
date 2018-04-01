from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dosnes import dosnes


X, y = datasets.load_digits(return_X_y = True)
metric = "sqeuclidean"

model = dosnes.DOSNES(metric = metric, verbose = 1, random_state=42)
X_embedded = model.fit_transform(X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y, cmap=plt.cm.Set1)
plt.title("Digits Dataset Embedded on a Sphere with metric {}".format(metric))
plt.show()
