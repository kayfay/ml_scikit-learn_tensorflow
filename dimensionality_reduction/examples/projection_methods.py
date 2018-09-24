# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
import os

# ML Imports
from sklearn.decomposition import PCA

# Graph Imports
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Directory Config
PROJECT_ROOT_DIR = '.'


# Declare functions
def image_path(fig_id):
    if not os.path.exists('images'):
        os.makedirs('images')
    return os.path.join(PROJECT_ROOT_DIR, 'images', fig_id)


def save_fig(fig_id, tight_layout=True):
    print("Saving ", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format="png", dpi=300)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._vert3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._vert3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# Build a 3D dataset
# Set stability for random generation measure
np.random.seed(42)
# Define parameters for model
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

# Define geometric state
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
# Instantiate empty matrix
X = np.empty((m, 3))
# Populate shapes across axes for data points
X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.rand(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

# Principal component analysis using singular value decomposition
# U \cdot \sum \cdot V^{T}
X_centered = X - X.mean(axis=0)  # Standardize the values
U, s, Vt = np.linalg.svd(X_centered)  # Use decomposition to return components
c1 = Vt.T[:, 0]  # Component 1
c2 = Vt.T[:, 1]  # Component 2

# Create a comparison matrix for dimensionality reduction verification
# Create a diagonal of the matrix of sums
m, n = X.shape
S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

# Test for accuracy of datasets
print("Are components similar", np.allclose(X_centered, U.dot(S).dot(Vt)))

# Set 2D from 3D PCA
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

X2D_using_svd = X2D

# PCA using scikit-learn
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

# PCA 2D subspace to 3D
X3D_inv = pca.inverse_transform(X2D)

# Compare
print("Scikit PCA reduced values\n", X2D[:5])
print("PCA with SVD\n", X2D_using_svd[:5])
print("Are PCA versions simliar", np.allclose(X2D, X2D_using_svd))

# Reconstruction error
print("Reconstruction error:{}".format(
    np.mean(np.sum(np.square(X3D_inv - X), axis=1))))

# Components

print("Indiviudal components from data", pca.components_,
      "\nVariance from data per component", pca.explained_variance_ratio_,
      "\nLoss of variance", pca.explained_variance_ratio_.sum())

# np.square(s) / np.square(s).sum() # using SVD

# Express the plane as a functoin of x and y
axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

# Plot the 3D dataset, the plane and the projections
fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')

X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]

ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
np.linalg.norm(C, axis=0)
ax.add_artist(
    Arrow3D(
        [0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]],
        mutation_scale=15,
        lw=1,
        arrowstyle="-|>",
        color="k"))
ax.add_artist(
    Arrow3D(
        [0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]],
        mutation_scale=15,
        lw=1,
        arrowstyle="-|>",
        color="k"))
ax.plot([0], [0], [0], "k.")

for i in range(m):
    if X[i, 2] > X3D_inv[i, 2]:
        ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]],
                [X[i][2], X3D_inv[i][2]], "k-")
    else:
        ax.plot(
            [X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]],
            [X[i][2], X3D_inv[i][2]],
            "k-",
            color="#505050")

ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])

save_fig("dataset_3d_plot")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

ax.plot(X2D[:, 0], X2D[:, 1], "k+")
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
ax.plot([0], [0], "ko")
ax.arrow(
    0,
    0,
    0,
    1,
    head_width=0.05,
    length_includes_head=True,
    head_length=0.1,
    fc="k",
    ec='k')
ax.arrow(
    0,
    0,
    1,
    0,
    head_width=0.05,
    length_includes_head=True,
    head_length=0.1,
    fc='k',
    ec='k')
ax.set_xlabel("$z_1$", fontsize=18)
ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
ax.axis([-1.5, 1.3, -1.2, 1.2])
ax.grid(True)

save_fig("dataset_2d_plot")
plt.show()
