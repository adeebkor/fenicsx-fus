import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("build/line_data.txt", delimiter=",")

unique_data = np.unique(data, axis=0)
unique_data = unique_data[unique_data[:, 0].argsort()]

x = unique_data[:, 0]
y = unique_data[:, 1]

plt.plot(x, y, '-')
plt.savefig("plot_over_line.png")
