// Copyright (C) 2022 Adeeb Arif Kor
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/finite-element.h>
#include <span>
#include <vector>

/// Reorder dofmap into tensor product order
/// @param[in] in_arr Input dofmap
/// @param[in] celltype Cell type
/// @param[in] p Degree of basis function
/// @param[out] out_arr Output dofmap
void reorder_dofmap(std::span<std::int32_t> out_arr, std::span<std::int32_t> in_arr,
                    basix::cell::type celltype, int p) {

  // Create element
  auto element = basix::create_element(basix::element::family::P, celltype, p,
                                       basix::element::lagrange_variant::gll_warped);

  // Get tensor product order
  auto [_, tensor_order] = element.get_tensor_product_representation()[0];

  int ndofs = tensor_order.size();
  int ncells = in_arr.size() / ndofs;

  // Reorder degrees of freedom into tensor product order
  for (int c = 0; c < ncells; ++c) {
    std::transform(tensor_order.begin(), tensor_order.end(), out_arr.begin() + c * ndofs,
                   [&](std::size_t i) { return in_arr[c * ndofs + i]; });
  }
}