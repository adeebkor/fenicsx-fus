import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data file
with open("test_eval_model3_p2.npy", "rb") as file:
    x = np.load(file)
    U = np.load(file)

print("Number of snapshot:", U.shape[0])
print(U.shape)
exit()
# Animate the figure
fig = plt.figure()

ims = []
for i in range(U.shape[0]):
    p = plt.plot(x[:, 0], U[i, :], animated=True, color='k')

    # plt.xlim([-0.01, 0.01])

    ims.append((p))

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

filename = "anim1d.mp4"

ani.save(filename,  fps=2, extra_args=['-vcodec', 'libx264'])

# plt.plot(x[:, 0], U[-10, :])
# plt.xlim([-0.01, 0.01])
# plt.show()