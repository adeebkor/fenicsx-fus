import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data file
with open("model3-1d.npy", "rb") as file:
    U = np.load(file)

print("Number of snapshot:", U.shape)

x0 = np.linspace(0, 0.1 * 4.5, 20000)

# Animate the figure
fig = plt.figure()

ims = []
for i in range(U.shape[0]):
    p = plt.plot(x0, U[i, :], animated=True, color='k')

    plt.xlim([0.1, 0.15])

    ims.append((p))

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

filename = "anim1d.mp4"

ani.save(filename,  fps=2, extra_args=['-vcodec', 'libx264'])
