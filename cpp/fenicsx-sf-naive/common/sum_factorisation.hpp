// Copyright (C) 2022 Igor A. Baratta, Adeeb Arif Kor
// SPDX-License-Identifier:    MIT

#pragma once

/// ------------------------------------------------------------------------ //
/// Transpose of the 2D tensor A and store in 2D tensor B
/// @param[in] A tensor
/// @param[out] B tensor
template <typename T, int Na, int Nb, int offa, int offb>
static inline void transpose(const T* __restrict__ A, T* __restrict__ B) {

  for (int a = 0; a < Na; ++a) {
    for (int b = 0; b < Nb; ++b) {
      B[a * offa + b * offb] = A[a * Nb + b];
    }
  }
}

/// ------------------------------------------------------------------------ //
/// Compute the tensor contraction C[a, b] = A[a, k] * B[k, c] as a
/// matrix-matrix multiplication
/// k is the contraction index
/// @param[in] A tensor of shape (Na, Nk)
/// @param[in] B tensor of shape (Nb, Nk) -> Shape (Nb, Nk) so that we can transverse row-wise
/// @param[out] C tensor of shape (Na, Nb)
template <typename T, int Na, int Nb, int Nk>
static inline void contract(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C) {

  for (int a = 0; a < Na; ++a) {
    for (int b = 0; b < Nb; ++b) {
      for (int k = 0; k < Nk; ++k) {
        C[a * Nb + b] += A[a * Nk + k] * B[b * Nk + k];
      }
    }
  }
}

/// ------------------------------------------------------------------------ //
/// Perform transpose of 3D tensor A and store in 3D tensor B
/// @param[in] A tensor of shape (Na, Nb, Nc)
/// @param[out] B output tensor
template <typename T, int Na, int Nb, int Nc, int offa, int offb, int offc>
static inline void transpose(T* __restrict__ A, T* __restrict__ B) {
  for (int a = 0; a < Na; a++)
    for (int b = 0; b < Nb; b++)
      for (int c = 0; c < Nc; c++)
        B[offa * a + offb * b + offc * c] = A[a * Nb * Nc + b * Nc + c];
}

/// ------------------------------------------------------------------------ //
/// Compute the tensor contraction C[{a, b}, c] = A[{a, b}, k] * B[k, c] as a
/// matrix-matrix multiplication
/// k is the contraction index
/// @param[in] A tensor of shape (Na, Nb, Nk)
/// @param[in] B tensor of shape (Nb, Nk) -> Shape (Nb, Nk) so that we can transverse row-wise
/// @param[out] C tensor of shape (Na, Nb, Nc)
template <typename T, int Na, int Nb, int Nc, int Nk>
static inline void contract(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C) {

  int Nd = Na * Nb;

  for (int d = 0; d < Nd; ++d) {
    for (int c = 0; c < Nc; ++c) {
      for (int k = 0; k < Nk; ++k) {
        C[d * Nc + c] += A[d * Nk + k] * B[c * Nk + k];
      }
    }
  }
}
