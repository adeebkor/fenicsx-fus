import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data file
data_fname = "test-model"
with open("{}.npy".format(data_fname), "rb") as file:
    U = np.load(file)

print("Number of snapshot:", U.shape)

# Set points
x0 = np.linspace(0, 0.45, 10000)

# Animate the figure
fig = plt.figure(figsize=(16,8))

ims = []
for i in range(U.shape[0]):
    p = plt.plot(x0, U[i, :], animated=True, color='k')

    plt.xlim([0.07, 0.14])

    ims.append((p))

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

anim_fname = "test-model"

ani.save("{}.mp4".format(anim_fname), fps=2, extra_args=['-vcodec', 'libx264'])
