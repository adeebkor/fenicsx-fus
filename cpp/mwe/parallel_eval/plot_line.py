import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("build/filename.txt", delimiter=",")

unique_data = np.unique(data, axis=0)
unique_data = unique_data[unique_data[:, 0].argsort()]

x = unique_data[:, 0]
y = unique_data[:, 1]

plt.plot(x, y, '-')
plt.savefig("pressure_on_line.png")
