import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import tqdm

from dosnes import dosnes

iris = datasets.load_iris()

X = iris.data
y = iris.target

metric = "sqeuclidean"

# df = pd.DataFrame({
#                 'momentum' : [],
#                 'final_momentum' : [],
#                 'mom_switch_iter' : [],
#                 'learning_rate' : [],
#                 'min_gain' : [],
#                 'iter': [],
#                 'cost' : [],
#                  })
#
# param_grid = {  'momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
#                 'final_momentum': [0.1, 0.3, 0.5, 0.7, 0.9],
#                 'mom_switch_iter': [150, 250, 400, 600],
#                 'learning_rate': [250, 500, 750],
#                 'min_gain': [0.001, 0.01, 0.1],
#             }
#
# for params in tqdm.tqdm(ParameterGrid(param_grid)):
#     model = dosnes.DOSNES(max_iter = 1000, verbose_freq = 10, metric = metric, verbose = 0, random_state=42, **params)
#     X_embedded = model.fit_transform(X, y)
#     content = []
#     for i, c in model.cost:
#         content.append([
#             params["momentum"],
#             params["final_momentum"],
#             params["mom_switch_iter"],
#             params["learning_rate"],
#             params["min_gain"],
#             i,
#             c
#         ])
#     df2 = pd.DataFrame(content, columns=['momentum', 'final_momentum', 'mom_switch_iter', 'learning_rate', 'min_gain', 'iter', 'cost'])
#     df = df.append(df2, ignore_index=True)
#
# df.to_csv("datas.csv")

momentum = 0.1
final_momentum = 0.7
mom_switch_iter = 250
max_iter = 1000
learning_rate = 400
min_gain = 0.01

model = dosnes.DOSNES(
    momentum = momentum, final_momentum = final_momentum, learning_rate = learning_rate, min_gain = min_gain,
    max_iter = 1000, verbose_freq = 10, metric = metric, verbose = 1, random_state=0)
X_embedded = model.fit_transform(X, y, filename="training.gif")

plt.plot(*zip(*model.cost))
plt.title("Evolution of the Cost function")
plt.ylabel("Cross-Entropy")
plt.xlabel("Iterations")
plt.savefig("cost.png")
plt.close('all')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y, cmap=plt.cm.Set1)
plt.title("Iris Dataset Embedded on a Sphere with metric {}".format(metric))
plt.show()