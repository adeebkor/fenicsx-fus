// Copyright (C) 2022 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <algorithm>
#include <vector>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <basix/mdspan.hpp>
#include <dolfinx.h>

namespace stdex = std::experimental;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;

using namespace dolfinx;

/// ---------------------------------------------------------------------------
/// Transpose matrix A and store in B
/// @param[in] A Input matrix
/// @param[out] B Output matrix
template <typename U, typename V>
void transpose(const U& A, V& B) {
  for (std::size_t i = 0; i < A.extent(0); ++i) {
    for (std::size_t j = 0; j < A.extent(1); ++j) {
      B(i, j) = A(j, i);
    }
  }
}

/// ---------------------------------------------------------------------------
/// Compute the scaled determinant of the Jacobian ([cell][point])
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The quadrature points to compute Jacobian of the map
/// @param[in] weights The weights evaluated at the quadrature points
std::vector<double> compute_scaled_jacobian_determinant(
  std::shared_ptr<const mesh::Mesh> mesh, std::vector<double> points,
  std::vector<double> weights)
{
  // Number of points
  std::size_t nq = weights.size();

  // Get geometry data
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
    = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const double> x_g = mesh->geometry().x();

  // Get dimensions
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t nc = mesh->topology().index_map(tdim)->size_local();

  // Tabulate basis functions at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<double> phi_b(
    std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  // Create working arrays
  std::vector<double> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

  std::vector<double> J_b(tdim * gdim);
  mdspan2_t J(J_b.data(), tdim, gdim);
  std::vector<double> detJ_b(nc * nq);
  mdspan2_t detJ(detJ_b.data(), nc, nq);
  std::vector<double> det_scratch(2 * tdim * gdim);

  for (std::size_t c = 0; c < nc; ++c)
  {
    // Get cell geometry (coordinates dofs)
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
    }

    // Compute the scaled Jacobian determinant
    for (std::size_t q = 0; q < nq; ++q)
    {
      std::fill(J_b.begin(), J_b.end(), 0.0);

      // Get the derivatives at each quadrature points
      auto dphi = stdex::submdspan(phi, std::pair(1, tdim+1), q, 
                                   stdex::full_extent, 0);

      // Compute Jacobian matrix
      auto _J = stdex::submdspan(J, stdex::full_extent, stdex::full_extent);
      cmap.compute_jacobian(dphi, coord_dofs, _J);

      // Compute the determinant of the Jacobian
      detJ(c, q) = cmap.compute_jacobian_determinant(_J, det_scratch);

      // Scaled the determinant of the Jacobian
      detJ(c, q) = std::fabs(detJ(c, q)) * weights[q];
    }
  }

  return detJ_b;
}

/// ---------------------------------------------------------------------------
/// Compute the scaled of the geometrical factor ([cell][points][tdim][gdim])
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The quadrature points to compute Jacobian of the map
/// @param[in] weights The weights evaluated at the quadrature points
std::vector<double> compute_scaled_geometrical_factor(
  std::shared_ptr<const mesh::Mesh> mesh, std::vector<double> points,
  std::vector<double> weights)
{
  // The number of element of the upper triangular matrix
  std::map<int, int> gdim2dim;
  gdim2dim[2] = 3;
  gdim2dim[3] = 6;

  // Number of points
  std::size_t nq = weights.size();

  // Get geometry data
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
    = mesh->geometry().dofmap();
  const std::size_t num_dofs_g = cmap.dim();
  std::span<const double> x_g = mesh->geometry().x();

  // Get dimensions
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->topology().dim();
  const std::size_t nc = mesh->topology().index_map(tdim)->size_local();
  int dim = gdim2dim[gdim];

  // Tabulate basis functions at quadrature points
  std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, nq);
  std::vector<double> phi_b(
    std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
  cmdspan4_t phi(phi_b.data(), phi_shape);
  cmap.tabulate(1, points, {nq, gdim}, phi_b);

  // Create working arrays
  std::vector<double> coord_dofs_b(num_dofs_g * gdim);
  mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

  // Jacobian
  std::vector<double> J_b(gdim * tdim);
  mdspan2_t J(J_b.data(), gdim, tdim);

  // Jacobian inverse J^{-1}
  std::vector<double> K_b(tdim * gdim);
  mdspan2_t K(K_b.data(), tdim, gdim);

  // Jacobian inverse transpose J^{-T}
  std::vector<double> KT_b(gdim * tdim);
  mdspan2_t KT(KT_b.data(), gdim, tdim);

  // G = J^{-1} * J^{-T}
  std::vector<double> G_b(gdim * tdim);
  mdspan2_t G(G_b.data(), gdim, tdim);

  // G small
  std::vector<double> Gs_b(nc * nq * dim);
  mdspan3_t Gs(Gs_b.data(), nc, nq, dim);

  // Jacobian determinants
  std::vector<double> detJ_b(nc * nq);
  mdspan2_t detJ(detJ_b.data(), nc, nq);
  std::vector<double> det_scratch(2 * gdim * tdim);

  for (std::size_t c = 0; c < nc; ++c)
  {
    // Get cell geometry (coordinates dofs)
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
    }

    // Compute the scaled geometrical factor
    for (std::size_t q = 0; q < nq; ++q)
    {
      std::fill(J_b.begin(), J_b.end(), 0.0);
      std::fill(K_b.begin(), K_b.end(), 0.0);
      std::fill(KT_b.begin(), KT_b.end(), 0.0);
      std::fill(G_b.begin(), G_b.end(), 0.0);
      
      // Get the derivatives at each quadrature points 
      auto dphi = stdex::submdspan(phi, std::pair(1, tdim+1), q,
                                   stdex::full_extent, 0);

      // Compute Jacobian matrix
      auto _J = stdex::submdspan(J, stdex::full_extent, stdex::full_extent);
      cmap.compute_jacobian(dphi, coord_dofs, _J);

      // Compute the inverse Jacobian matrix
      auto _K = stdex::submdspan(K, stdex::full_extent, stdex::full_extent);
      cmap.compute_jacobian_inverse(_J, _K);

      // Transpose K -> K^{T}
      auto _KT = stdex::submdspan(KT, stdex::full_extent, stdex::full_extent);
      transpose(_K, _KT);

      // Compute the scaled geometrical factor (K * K^{T})
      auto _G = stdex::submdspan(G, stdex::full_extent, stdex::full_extent);
      math::dot(_K, _KT, _G);

      // Compute the scaled Jacobian determinant
      detJ(c, q) = cmap.compute_jacobian_determinant(_J, det_scratch);
      detJ(c, q) = std::fabs(detJ(c, q)) * weights[q];

      // Only store the upper triangular values since G is symmetric
      for (std::size_t a = 0; a < gdim; ++a) {
        for (std::size_t b = 0; b < a+1; ++b) {
          Gs(c, q, a+b) = detJ(c, q) * G(a, b);
        }
      }
    }
  }
  return Gs_b;
}

/// ---------------------------------------------------------------------------
/// Tabulate degree P basis functions on an interval
std::vector<double> tabulate_1d(int P, int Q, int derivative)
{
  // Create element
  auto element = basix::create_element(
    basix::element::family::P, basix::cell::type::interval, P, 
    basix::element::lagrange_variant::gll_warped);
  
  // Create quadrature
  auto [points, weights] = basix::quadrature::make_quadrature(
    basix::quadrature::type::gll, basix::cell::type::interval, Q);

  // Tabulate
  auto [table, shape] = element.tabulate(1, points, {weights.size(), 1});

  return table;

}

/*

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

/// ---------------------------------------------------------------------------
/// Compute the Jacobian of coordinate transformation for all cells in the mesh 
/// given some points in the reference domain.
/// @param[in] mesh The mesh object (which contains the coordinate map)
/// @param[in] points The points where to compute jacobian of the map
/// @return The jacobian for all cells in the computed at quadrature points
std::vector<double> compute_jacobian(std::shared_ptr<const mesh::Mesh> mesh) 
{
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
}

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
}
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
}
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

*/
