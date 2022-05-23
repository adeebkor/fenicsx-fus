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
//   std::array<double, M*N> _phi = {1, 0, 0, 1};

  // Fill the tensor with values from 0 to N*N*N
  for (int i = 0; i < N*N*N; i++) {
    _x[i] = i + 1;
  }

  // Print the tensor
  double* xi = _x.data();
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
  for (int i = 0; i < M*N; i++) {
    std::cout << *(dphi + i) << " ";
  }
  std::cout << "\n";

//   double* phi = _phi.data();
//   for (int i = 0; i < M*N; i++) {
//     std::cout << *(phi + i) << " ";
//   }
//   std::cout << "\n";


//   Buffer<double, M, N> buffer;
//   buffer.zero();

  std::array<double, M*N*N> out{0}, outT{0};
  contract<double, N, M, N, N, true>(dphi, xi, out.data());
  transpose<double, N, M, N, M*N, N, 1>(out.data(), outT.data());
//   apply_contractions<double, M, N, true>(dphi, phi, phi, xi, out.data(), buffer);

  for (int i = 0; i < M*N*N; i++) {
    std::cout << outT[i] << " ";
  }
  std::cout << "\n";

//   buffer.zero();
//   std::array<double, M*M*M> out0;

  // x-contraction
//   contract<double, N, M, N, N, true>(dphi, xi, buffer.T0.data());
//   transpose<double, M, N, N, M*N, N, 1>(buffer.T0.data(), out0.data());

  // y-contraction
//   transpose<double, N, N, N, N, N*N, 1>(xi, buffer.T0.data());
//   contract<double, N, M, N, N, true>(dphi, buffer.T0.data(), buffer.T1.data());
//   transpose<double, N, N, N, N, N*N, 1>(buffer.T1.data(), out0.data());

//   for (int i = 0; i < M*M*M; i++) {
//     std::cout << out0[i] << " ";
//   }
//   std::cout << "\n";

  return 0;
}