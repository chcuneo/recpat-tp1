import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Multivariate normal 1 mean
mean_1 = [-2.5, -1.5]
# Multivariate normal 1 covariance matrix
sigma_1=3.0
sigma_2=3.0
rho=-0.8
cov_1 = [[sigma_1**2, rho*sigma_1*sigma_2], [rho*sigma_1*sigma_2, sigma_2**2]]
gd1 = multivariate_normal(mean=mean_1, cov=cov_1)

# Multivariate normal 2 mean
mean_2 = [2, 2]
# Multivariate normal 2 covariance matrix
cov_2 = [[10, 5], [5, 5]]  # diagonal covariance
gd2 = multivariate_normal(mean=mean_2, cov=cov_2)

# Generate grid of poins in 2d space, to sample the probility density of both bivariate gaussians
x = y = np.arange(-10.0, 10.0, 0.05)
X, Y = np.meshgrid(x, y)

# Sample the densities
zs = np.array([max(gd1.pdf([x,y]), gd2.pdf([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

# Graph the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
plt.title('Probability Density')

ax.plot_surface(X, Y, Z, color='b', vmin=0, vmax=0.03, cmap='RdYlGn_r')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Probability Density')

plt.savefig('ej1-density.png')

# Graph bayesian decision classifier boundaries and both bivariates gaussians regions
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
plt.title('Areas')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

areas = ax.pcolormesh(X, Y, np.array([-1 if gd1.pdf([x,y]) > gd2.pdf([x,y]) else 1 for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape), alpha=0.13, vmin=-1, vmax=1, cmap='RdBu')
cf1 = ax.contour(X, Y, np.array([gd1.pdf([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape), alpha=0.5, vmin=-0.03, vmax=0.03, cmap='Reds')
cf2 = ax.contour(X, Y, np.array([gd2.pdf([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape), alpha=0.5, vmin=-0.03, vmax=0.03, cmap='Blues')

plt.savefig('ej1-area.png')
