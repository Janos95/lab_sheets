import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
import numpy as np

_COLORS = np.load(path.join(path.dirname(__file__), 'pca_toy_colors.npy'))


def plot_label_3d(x, y):
    """
    Show a 3D scatter plot of data.

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


def plot_label_2d_impl(x, y):
    x = np.array(x)
    y = np.array(y).reshape(-1).astype(int)
    if len(x.shape) != 2 or len(y) != x.shape[0] or x.shape[1] != 2:
        raise ValueError("Invalid input shape")
    plt.scatter(x[:,0], x[:,1], c=_COLORS[y], s=30)
    plt.gca().set_aspect('equal')

def plot_label_2d(x,y):
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
    plot_label_2d_impl(x,y)    
    plt.show()


def PlotContourLine(func, value=0, minx=0, maxx=10, miny=0, maxy=10, x=None, y=None):
    #This plots the contourline func(x) = value
    
    samplenum = 1000
    xrange = np.arange(minx, maxx, (maxx-minx)/samplenum)
    yrange = np.arange(miny, maxy, (maxy-miny)/samplenum)
    
    #This generates a two-dimensional mesh
    X, Y = np.meshgrid(xrange,yrange)
    
    argsForf = np.array([X.flatten(),Y.flatten()]).T
    Z = func(argsForf)
    Z = np.reshape(Z,X.shape)
    
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    #plt.contour(X, Y, Z, alpha=0.5,levels=[value],linestyles='dashed',linewidths=3)
    Z = np.where(Z > value, 1, -1)
    plt.contourf(X, Y, Z, alpha=0.2, colors=_COLORS[:2])
    if not x is None and not y is None:
        plot_label_2d_impl(x,y)
    plt.show()
    
    
    
    
    
    
