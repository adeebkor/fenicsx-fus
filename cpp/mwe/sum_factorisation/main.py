import numpy as np

M = 3
N = 2

xi = np.arange(N*N*N).reshape(N, N, N)
phi = np.arange(M*N).reshape(M, N)

T = np.einsum('iq, qjk', phi, xi)

print("T = \n", T)
print("T transpose [2, 0, 1] = \n",  np.transpose(T, [2, 0, 1]))
