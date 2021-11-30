#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/math.h>

using namespace dolfinx;

// Compute C = op(A)*op(B)
// op = T() if transpose is true
// Assumes that ldA and ldB are 3
template <typename U, typename V, typename P>
void dot(const U& A, const V& B, P& C, bool transpose = false) {
  constexpr int ldA = 2;
  constexpr int ldB = 2;

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

xt::xtensor<double, 2> precompute_jacobian(std::shared_ptr<const mesh::Mesh> mesh, int q) {
  common::Timer t("~precompute detJ");
  // Tabulate quadrature points and weights
  auto cell = basix::cell::type::quadrilateral;
  auto quad = basix::quadrature::type::gll;
  auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);
  const std::size_t nq = weights.size();

  const mesh::Geometry& geometry = mesh->geometry();
  const mesh::Topology& topology = mesh->topology();
  const fem::CoordinateElement& cmap = geometry.cmap();

  const std::size_t tdim = topology.dim();
  const std::size_t gdim = geometry.dim();
  const std::size_t ncells = mesh->topology().index_map(tdim)->size_local();

  // Tabulate coordinate map basis functions
  xt::xtensor<double, 4> basis = cmap.tabulate(1, points);
  xt::xtensor<double, 2> phi = xt::view(basis, 0, xt::all(), xt::all(), 0);
  xt::xtensor<double, 3> dphi = xt::view(basis, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

  const xt::xtensor<double, 2>& x = geometry.x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_nodes = x_dofmap.num_links(0);

  xt::xtensor<double, 2> J = xt::zeros<double>({tdim, gdim});
  xt::xtensor<double, 2> coords = xt::zeros<double>({num_nodes, gdim});
  xt::xtensor<double, 2> dphi_q = xt::zeros<double>({tdim, num_nodes});
  xt::xtensor<double, 2> detJ({ncells, nq});

  // Compute determinant of the coordinates
  for (std::size_t c = 0; c < ncells; c++) {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i) {
      std::copy_n(xt::row(x, x_dofs[i]).begin(), 3, xt::row(coords, i).begin());
    }

    for (std::size_t q = 0; q < nq; q++) {
      dphi_q = xt::view(dphi, xt::all(), q, xt::all());
      J.fill(0);
      dot(coords, dphi_q, J, true);
      detJ(c, q) = std::fabs(math::det(J)) * weights[q];
    }
  }

  return detJ;
}

// Tabulate order P basis functions on a Hexahedron
xt::xtensor<double, 2> tabulate_basis(int p, int q) {
  // Tabulate quadrature points and weights
  auto family = basix::element::family::P;
  auto cell = basix::cell::type::quadrilateral;
  auto quad = basix::quadrature::type::gll;
  auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);
  auto variant = basix::element::lagrange_variant::gll_warped;
  auto element = basix::create_element(family, cell, p, variant);
  xt::xtensor<double, 4> basis = element.tabulate(0, points);
  return xt::view(basis, 0, xt::all(), xt::all(), 0);
}