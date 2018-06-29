from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
import numpy as np
from scipy.io import loadmat
from scipy.special import ellipeinc, ellipe
from os import path



def plot_toy_3d(x, y):
    """
    Show a 3D scatter plot of our toy data.

    Parameters
    ----------

    x: (n, 3) array-like
    y: (n,) array-like
        The labels
    """
    y = y.astype(int)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], x[:,2], c=_COLORS[y])
    ax.set_aspect('equal')
    plt.show()


def plot_toy_2d(x, y):
    """
    Plot the low-dimensional representation of the toy data.

    The function uses a colored scatter plot where each point is
    from `x` and the color is determined by the labels `y`.

    Parameters
    ----------
    
    x: (n, 2) array-like
        The two dimensional representation.
    y: (n,) array-like
       The labels.
    """
    x = np.array(x)
    y = np.array(y).reshape(-1).astype(int)
    if len(x.shape) != 2 or len(y) != x.shape[0] or x.shape[1] != 2:
        raise ValueError("Invalid input shape")
    plt.scatter(x[:,0], x[:,1], c=_COLORS[y], s=30)
    plt.gca().set_aspect('equal')
    plt.show()




_COLORS = np.load(path.join(path.dirname(__file__), 'pca_toy_colors.npy'))


