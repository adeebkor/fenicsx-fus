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
void reorder_dofmap(
  std::span<std::int32_t> out_arr,
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<const std::int32_t, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>> in_arr, 
  basix::cell::type celltype, int p) {

  // Tensor product ordering
  auto tp_order = basix::tp_dof_ordering(
    basix::element::family::P, celltype, p,
    basix::element::lagrange_variant::gll_warped,
    basix::element::dpc_variant::unset, false);

  // Index sort
  std::vector<std::int32_t> perm(tp_order.size());
  std::iota(perm.begin(), perm.end(), 0);

  std::sort(perm.begin(), perm.end(),
            [&tp_order](std::int32_t i1, std::int32_t i2) 
            {return tp_order[i1] < tp_order[i2]; });
    
  int Nd = tp_order.size();  // Number of DOF
  int Nc = in_arr.size() / Nd;  // Number of cells

  // Reorder degrees of freedom into tensor product order
  for (int c = 0; c < Nc; ++c) {
    std::transform(perm.begin(), perm.end(), out_arr.begin() + c * Nd,
                   [&](std::int32_t idx) { return in_arr(c, idx); });
  }
}