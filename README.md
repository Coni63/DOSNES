# Doubly Stochastic Neighbor Embedding on Spheres

The repository propose a Python implementation of the Doubly Stochastic Neighbor Embedding on Spheres (DOSNES) published by Yao Lu, Zhirong Yang, Jukka Corander on [Arxiv](https://arxiv.org/abs/1609.01977) in Sep. 2016. It is based on the Matlab implemantion available on [Github](https://github.com/yaolubrain/DOSNES). 

The principle of this model is to embed an high dimensionnal array on a 3D Sphere. The principle is the same as the TSNE but at every iteration, all embedded points are forced to be on a sphere. 

![rendering](https://github.com/Coni63/DOSNES/blob/master/images/iris.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

No prerequisites are needed, you can download the repository or clone it. 

```
git clone https://github.com/Coni63/DOSNES.git
```

### Installing

No pip installation is available, you just have to include the package in you project folder.


## How to use

The model has been build to be similar to sklearn model. As imple example is available below or in test_XX.py files.

```python
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
```

You can provide either a distance matrix custom if you set the metric to pre-computed, or provide you dataset and the metric to use (must be part of distances available with the function [pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) of scipy)


## Contributor

* **Nicolas MINE** - *Initial work* - [Coni63](https://github.com/Coni63)
* **btaba** - Implementation of Sinkhorn Knopp Algorithm - [btaba](https://github.com/btaba/sinkhorn_knopp)
* **Paul Panzer** - *Support on StackOverflow*


## Acknowledgments

There may be still some errors. For example on digit dataset, the result is not as good as the one from the official paper of matlab implementation.
There is no checks / error handling implemented yet.
You should not provide a dataset with missing values. 