#pragma once

#include "precomputation.hpp"
#include <cstdint>
#include <map>
#include <cmath>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>


std::pair<std::vector<int>, xt::xtensor<double, 4>>
tabulate_basis_and_permutation(int p, int q){
  // Tabulate quadrature points and weights
  auto family = basix::element::family::P;
  auto cell = basix::cell::type::quadrilateral;
  auto quad = basix::quadrature::type::gll;
  auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);
  auto variant = basix::element::lagrange_variant::gll_warped;
  auto element = basix::create_element(family, cell, p, variant);

  xt::xtensor<double, 4> table = element.tabulate(1, points);
  std::vector<int> perm = std::get<1>(element.get_tensor_product_representation()[0]);

  // Clamp -1, 0, 1 values
  xt::filtration(table, xt::isclose(table, -1.0)) = -1.0;
  xt::filtration(table, xt::isclose(table, 0.0)) = 0.0;
  xt::filtration(table, xt::isclose(table, 1.0)) = 1.0;

  return {perm, table};
}

template <typename T>
class 
