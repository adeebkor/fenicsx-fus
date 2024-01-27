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

template <typename T, int Na, int Nb>
struct Buffer {
  std::array<T, Na * Nb * Nb> T0{0};
  std::array<T, Na * Nb * Nb> T0_t{0};
  std::array<T, Na * Na * Nb> T1{0};
  std::array<T, Na * Na * Nb> T1_t{0};
  std::array<T, Na * Na * Na> T2{0};

  void zero() {
    T0 = {0};
    T1 = {0};
    T2 = {0};
  }
};

// --------------------------------------------------------------------//
/// Compute the tensor contraction C[a,b,c] <- A[a, k] B[k, b, c]
/// as a matrix- matrix multiplication C[a,{b, c}] = A[a, k] B[k, {b, c}]
/// K is the contraction index
template <typename T, int Nk, int Na, int Nb, int Nc, bool transpose>
static inline void contract(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C) {
  // d = {b, c} is a multi-index d = b* Nc + c
  constexpr int Nd = Nb * Nc;

  if constexpr (transpose) {
    for (int k = 0; k < Nk; k++)
      for (int a = 0; a < Na; a++)
        for (int d = 0; d < Nd; d++)
          C[a * Nd + d] += A[a * Nk + k] * B[k * Nd + d];
  } else {
    for (int k = 0; k < Nk; k++)
      for (int a = 0; a < Na; a++)
        for (int d = 0; d < Nd; d++)
          C[a * Nd + d] += A[k * Na + a] * B[k * Nd + d];
  }
}

// --------------------------------------------------------------------//
/// Compute A = [I0 x I1 x I2]  B by successive tensor contractions
/// A[Na, Na, Na]
/// In[Na, Nb]
/// B[Nb, Nb, Nb]
template <typename T, int Na, int Nb, bool Tr>
static inline void apply_contractions(const T* I0, const T* I1, const T* I2, T* __restrict__ B,
                                      T* __restrict__ A, Buffer<T, Na, Nb>& buffer) {
  buffer.zero();
  // First tensor contraction
  // T0[Na, Nb, Nb] <- I0[Na, Nb] B[Nb, Nb, Nb]
  // [a0, b1, b2] <- I0[a0, b0] B[b0, b1, b2]
  contract<T, Nb, Na, Nb, Nb, Tr>(I0, B, buffer.T0.data());

  // Transpose tensor, so the index of contraction appears first (ik = b1)
  // [b1, a0, b2] <- [a0, b1, b2]
  transpose<T, Na, Nb, Nb, Nb, Nb * Na, 1>(buffer.T0.data(), buffer.T0_t.data());

  // Second tensor contraction
  // T1[Na, Na, Nb] <- I1[Na, Nb] T0[Nb, Na, Nb]
  // [a1, a0, b2] <- [a1, b1] [b1, a0, b2]
  contract<T, Nb, Na, Na, Nb, Tr>(I1, buffer.T0_t.data(), buffer.T1.data());

  // Transpose tensor, so the index of contraction appears first (ik = b2)
  // [b2, a0, a1] <- [a1, a0, b2]
  transpose<T, Na, Na, Nb, 1, Na, Na * Na>(buffer.T1.data(), buffer.T1_t.data());

  // Third tensor contraction
  // T2[Na, Na, Na] <- phi[Na, Nb] T1[Nb, Na, Na]
  // [a2, a0, a1] <- [a2, b2][b2, a0, a1]
  contract<T, Nb, Na, Na, Na, Tr>(I2, buffer.T1_t.data(), buffer.T2.data());

  // Transpose tensor
  // [a0, a1, a2] <- [a2, a0, a1]
  transpose<T, Na, Na, Na, 1, Na * Na, Na>(buffer.T2.data(), A);
}