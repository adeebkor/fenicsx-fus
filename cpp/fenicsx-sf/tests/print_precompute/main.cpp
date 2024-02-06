// ====================
// Print geometric data
// ====================
// Copyright (C) Adeeb Arif Kor

#include "precompute.hpp"

#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;
using T = float;

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    // Create mesh
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        mesh::create_box<T>(MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
                                  {8, 8, 8}, mesh::CellType::hexahedron, part));

    // Create a map between polynomial degree and basix quadrature degree
    std::map<int, int> Qdegree;
    Qdegree[2] = 3;
    Qdegree[3] = 4;
    Qdegree[4] = 6;
    Qdegree[5] = 8;
    Qdegree[6] = 10;
    Qdegree[7] = 12;
    Qdegree[8] = 14;
    Qdegree[9] = 16;
    Qdegree[10] = 18;

    int P = 3;
    int Q = Qdegree[P]; 
    auto [pts, wts] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::gll, basix::cell::type::hexahedron, 
      basix::polyset::type::standard, Q);
    std::size_t nq = wts.size();

    auto detJ = compute_scaled_jacobian_determinant<T>(mesh, pts, wts);

    const std::size_t tdim = mesh->topology()->dim();
    const std::size_t nc = mesh->topology()->index_map(tdim)->size_local();

    for (std::size_t c = 0; c < 1; ++c) {
      for (std::size_t q = 0; q < wts.size(); ++q) {
        std::cout << detJ[q + c*nq] << ", ";
      }
      std::cout << "\n";
    }

    auto G = compute_scaled_geometrical_factor<T>(mesh, pts, wts);

    // for (std::size_t c = 0; c < 1; ++c) {
    //   for (std::size_t q = 0; q < 2; ++q) {
    //     for (std::size_t i = 0; i < 6; ++i){
    //       std::cout << i + q*6 + c*6*nq << ": " << G[i + q*6 + c*6*nq] << "\n";
    //     }
    //   }
    // }
  }
  PetscFinalize();
}