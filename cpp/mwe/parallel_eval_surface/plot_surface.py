import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

data = np.genfromtxt("build/surface_data.txt", delimiter=",")

unique_data = np.unique(data, axis=0)
unique_data = unique_data[unique_data[:, 0].argsort()]

x = unique_data[:, 0]
y = unique_data[:, 1]
z = unique_data[:, 2]

xi = np.linspace(-1, 1, 100)
yi = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(xi, yi)
Z = griddata((x, y), z, (X, Y), method="nearest")

plt.contourf(X, Y, Z)
plt.savefig("plot_over_surface.png")
