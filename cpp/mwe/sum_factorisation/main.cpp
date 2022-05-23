#include "form.h"

#include <iostream>
#include <cmath>
#include <array>

#include "sum_factorisation.hpp"

int main(int argc, char* argv[]) {
  const int M = 3;
  const int N = 2;

  std::array<double, N*N*N> _x;
  std::array<double, M*N> _dphi;

  // Fill the tensor with values from 0 to N*N*N
  for (int i = 0; i < N*N*N; i++) {
    _x[i] = i;
  }

  // Print the tensor
  double* xi = _x.data();
  std::cout << "x = ";
  for (int i = 0; i < N*N*N; i++) {
    std::cout << *(xi + i) << " ";
  }
  std::cout << "\n";

  // Fill the matrix with values from 0 to M*N
  for (int i = 0; i < M*N; i++) {
    _dphi[i] = i;
  }

  // Print the matrix
  double* dphi = _dphi.data();
  std::cout << "dphi = ";
  for (int i = 0; i < M*N; i++) {
    std::cout << *(dphi + i) << " ";
  }
  std::cout << "\n";

  // Perform contraction
  std::array<double, M*N*N> _out{0};
  contract<double, N, M, N, N, true>(dphi, xi, _out.data());

  // Print output of contraction
  std::cout << "out = ";
  for (int i = 0; i < M*N*N; i++) {
    std::cout << _out[i] << " ";
  }
  std::cout << "\n";

  // Perform transpose (ijk => kji)
  std::array<double, M*N*N> _out_t{0};
  transpose<double, M, N, N, 1, M, M*N>(_out.data(), _out_t.data());

  // Print transpose of output
  std::cout << "out^{T} = ";
  for (int i = 0; i < N*N*M; i++) {
    std::cout << _out_t[i] << " ";
  }
  std::cout << "\n";

  return 0;
}