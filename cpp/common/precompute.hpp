// Copyright (C) 2022 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/math.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>

using namespace dolfinx;

/// ------------------------------------------------
// Compute C = op(A)*op(B)
// op = T() if transpose is true
// Assumes that ldA and ldB are 3
template <typename U, typename V, typename P>
void dot(const U& A, const V& B, P& C, bool transpose = false) {
  constexpr int ldA = 3;
  constexpr int ldB = 3;

  if (transpose) {
    const int num_nodes = A.shape(0);
    for (int i = 0; i < ldA; i++) {
      for (int j = 0; j < ldB; j++) {
        for (int k = 0; k < num_nodes; k++) {
          C(i, j) += A(k, i) * B(j, k);
        }
      }
    }
  } else {
    const int num_nodes = A.shape(1);
    for (int i = 0; i < ldA; i++) {
      for (int j = 0; j < ldB; j++) {
        for (int k = 0; k < num_nodes; k++) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }
  }
}

/// ------------------------------------------------
/// Compute the Jacobian of coordinate transformation for all cells in the mesh given
/// some points in the reference domain.
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The points where to compute jacobian of the map
/// @return The jacobian for all cells in the computed at quadrature points
xt::xtensor<double, 4> compute_jacobian(std::shared_ptr<const mesh::Mesh> mesh,
                                        xt::xtensor<double, 2> points) {
  // Number of quadrature points
  const std::size_t nq = points.shape(0);

  const mesh::Geometry& geometry = mesh->geometry();
  const mesh::Topology& topology = mesh->topology();
  const fem::CoordinateElement& cmap = geometry.cmap();

  const std::size_t tdim = topology.dim();
  const std::size_t gdim = geometry.dim();
  const std::size_t ncells = mesh->topology().index_map(tdim)->size_local();

  // Tabulate coordinate map basis functions
  xt::xtensor<double, 4> basis = cmap.tabulate(1, points);
  xt::xtensor<double, 2> phi = xt::view(basis, 0, xt::all(), xt::all(), 0);
  xt::xtensor<double, 3> dphi
      = xt::view(basis, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

  // Get geometry data
  xtl::span<const double> x = geometry.x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  // FIXME: Assumes one cell type
  const std::size_t num_nodes = x_dofmap.num_links(0);

  xt::xtensor<double, 4> J = xt::zeros<double>({ncells, nq, tdim, gdim});
  xt::xtensor<double, 2> coords = xt::zeros<double>({num_nodes, gdim});
  xt::xtensor<double, 2> dphi_q = xt::zeros<double>({tdim, num_nodes});

  // Compute Jacobian at quadratre points for all cells in the mesh
  for (std::size_t c = 0; c < ncells; c++) {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);

    // Copying x to coords
    for (std::size_t i = 0; i < x_dofs.size(); i++)
      common::impl::copy_N<3>(std::next(x.begin(), 3 * x_dofs[i]),
                              std::next(coords.begin(), 3 * i));

    for (std::size_t q = 0; q < nq; q++) {
      dphi_q = xt::view(dphi, xt::all(), q, xt::all());
      auto _J = xt::view(J, c, q, xt::all(), xt::all());
      // small gemm
      dot(coords, dphi_q, _J, true);
    }
  }
  return J;
}

/// ------------------------------------------------
/// Compute the determinant of the Jacobian (4d x [cell][point][tdim][gdim])
/// @param[in] J The jacobian
/// @return The determinant of the jacobian [ncells]x[npoints]
xt::xtensor<double, 2> compute_jacobian_determinant(xt::xtensor<double, 4>& J) {
  const std::size_t ncells = J.shape(0);
  const std::size_t npoints = J.shape(1);

  xt::xtensor<double, 2> detJ = xt::empty<double>({ncells, npoints});

  for (std::size_t c = 0; c < ncells; c++) {
    for (std::size_t q = 0; q < npoints; q++) {
      auto _J = xt::view(J, c, q, xt::all(), xt::all());
      detJ(c, q) = dolfinx::math::det(_J);
    }
  }
  return detJ;
};

/// ------------------------------------------------
/// Compute the inverse of the Jacobian (4d x [cell][point][tdim][gdim])
/// @param[in] J The jacobian
/// @return The inverse of the jacobian [ncells]x[npoints]x[gdim]x[tdim]
xt::xtensor<double, 4> compute_jacobian_inverse(xt::xtensor<double, 4>& J) {
  const std::size_t ncells = J.shape(0);
  const std::size_t npoints = J.shape(1);
  const std::size_t tdim = J.shape(2);
  const std::size_t gdim = J.shape(3);

  // Allocate data for inverse matrix
  // Note that gdim and tdim are swapped
  xt::xtensor<double, 4> K = xt::empty<double>({ncells, npoints, gdim, tdim});

  // For each cell, each quadrature point, compute inverse
  for (std::size_t c = 0; c < ncells; c++) {
    for (std::size_t q = 0; q < npoints; q++) {
      auto _J = xt::view(J, c, q, xt::all(), xt::all());
      auto _K = xt::view(K, c, q, xt::all(), xt::all());
      dolfinx::math::inv(_J, _K);
    }
  }
  return K;
};
/// ------------------------------------------------
/// Compute the inverse of the geometrical factor (4d x [cell][point][tdim][gdim])
/// @param[in] J The jacobian
/// @return The inverse of the jacobian [ncells]x[npoints]x[gdim]x[tdim]
xt::xtensor<double, 3> compute_geometrical_factor(xt::xtensor<double, 4>& J,
                                                  xt::xtensor<double, 2>& detJ,
                                                  std::vector<double>& weights) {

  const std::size_t ncells = J.shape(0);
  const std::size_t npoints = J.shape(1);
  const std::size_t tdim = J.shape(2);
  const std::size_t gdim = J.shape(3);
  const std::size_t dim = 6;

  xt::xtensor<double, 4> G_ = xt::zeros<double>({ncells, npoints, gdim, tdim});
  xt::xtensor<double, 3> G = xt::zeros<double>({ncells, npoints, dim});
  xt::xtensor<double, 2> K = xt::empty<double>({gdim, tdim});
  xt::xtensor<double, 2> KT = xt::empty<double>({gdim, tdim});

  // For each cell, each quadrature point, compute inverse
  for (std::size_t c = 0; c < ncells; c++) {
    for (std::size_t q = 0; q < npoints; q++) {
      double _detJ = detJ(c, q) * weights[q];
      K.fill(0);
      auto _J = xt::view(J, c, q, xt::all(), xt::all());
      auto _G = xt::view(G_, c, q, xt::all(), xt::all());
      auto g = xt::view(G, c, q, xt::all());
      dolfinx::math::inv(_J, K);
      KT = xt::transpose(K);
      dot(K, KT, _G);
      _G = _G * _detJ;

      // Only store the upper triangular values since G is symmetric
      g[0] = xt::view(_G, 0, 0); // G[0, 0]
      g[1] = xt::view(_G, 0, 1); // G[0, 1] = G[1, 0]
      g[2] = xt::view(_G, 0, 2); // G[0, 2] = G[2, 0]
      g[3] = xt::view(_G, 1, 1); // G[1, 1]
      g[4] = xt::view(_G, 1, 2); // G[1, 2] = G[2, 1]
      g[5] = xt::view(_G, 2, 2); // G[2, 2]

    }
  }
  return G;
};
/// ------------------------------------------------
// Tabulate order P basis functions on an interval
xt::xtensor<double, 2> tabulate_1d(int p, int q, int derivative) {
  // Tabulate quadrature points and weights
  auto family = basix::element::family::P;
  auto cell = basix::cell::type::interval;
  auto quad = basix::quadrature::type::gll;
  auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);
  auto variant = basix::element::lagrange_variant::gll_warped;
  auto element = basix::create_element(family, cell, p, variant);
  xt::xtensor<double, 4> basis = element.tabulate(derivative, points);
  
  // Clamped values
  xt::filtration(basis, xt::isclose(basis, -1.0)) = -1.0;
  xt::filtration(basis, xt::isclose(basis, 0.0)) = 0.0;
  xt::filtration(basis, xt::isclose(basis, 1.0)) = 1.0;
  
  return xt::view(basis, derivative, xt::all(), xt::all(), 0);
}

/// ------------------------------------------------
std::vector<int> compute_permutations(int p) {
  auto family = basix::element::family::P;
  auto cell = basix::cell::type::hexahedron;
  auto variant = basix::element::lagrange_variant::gll_warped;
  auto element = basix::create_element(family, cell, p, variant);
  auto [elements, perm] = element.get_tensor_product_representation()[0];
  return perm;
}

/// ------------------------------------------------
template <typename T>
xt::xtensor<T, 3> assemble_element_tensor(
    std::shared_ptr<fem::Form<T>> a,
    const std::vector<std::shared_ptr<const fem::DirichletBC<T>>>& bcs) {
  if (a->rank() != 2)
    throw std::runtime_error("Form rank should be 2.");

  auto mesh = a->mesh();
  assert(mesh);
  int dim = mesh->topology().dim();
  std::int32_t ncells = mesh->topology().index_map(dim)->size_local();
  auto dofmap = a->function_spaces()[0]->dofmap()->list();
  std::int32_t ndofs = dofmap.num_links(0);

  xt::xtensor<T, 3> A = xt::zeros<T>({ncells, ndofs, ndofs});
  int cell = 0;
  std::function<int(std::int32_t, const std::int32_t*, std::int32_t, const std::int32_t*,
                    const T*)>
      add = [&](int32_t nr, const int32_t* rows, int32_t nc, const int32_t* cols,
                const T* vals) {
        for (int i = 0; i < nr; i++)
          for (int j = 0; j < nc; j++)
            A(cell, i, j) += vals[i * nc + j];
        cell++;
        return 0;
      };

  fem::assemble_matrix(add, *a, bcs);

  return A;
}