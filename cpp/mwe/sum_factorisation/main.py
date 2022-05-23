import numpy as np

M = 3
N = 2

xi = np.arange(N*N*N).reshape(N, N, N)
phi = np.arange(M*N).reshape(M, N)

T = np.einsum('iq, qjk', phi, xi)

# print(T)
# print(T.flatten())
print(np.transpose(T, [2, 0, 1]))

# T_t0 = np.transpose(T, [0, 2, 1])
# T_t1 = np.transpose(T_t0, [2, 1, 0])

# print(T_t1)
