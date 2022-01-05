#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/math.h>
#include <xtensor/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>


std::pair<xt::xtensor<double, 4>, xt::xtensor<double, 2>>
precompute_jacobian_data(std::shared_ptr<const mesh::Mesh> mesh, int q){
	// Get geometrical and topological data
	const mesh::Geometry& geometry = mesh->geometry();
	const mesh::Topology& topology = mesh->topology();
	const fem::CoordinateElement& cmap = geometry.cmap();

	const std::size_t tdim = topology.dim();
	const std::size_t gdim = geometry.dim();
	const std::size_t ncells = mesh->topology().index_map(tdim)->size_local();

	const xt::xtensor<double, 2> x = xt::adapt(geometry.x().data(), geometry.x().size(), xt::no_ownership(), std::vector{geometry.x().size() / 3, std::size_t(3)});
	const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
	const std::size_t num_nodes = x_dofmap.num_links(0);

	// Tabulate quadrature points and weights
	auto cell = basix::cell::type::quadrilateral;
	auto quad = basix::quadrature::type::gll;
	auto [points, weights] = basix::quadrature::make_quadrature(quad, cell, q);
	const std::size_t nq = weights.size();

	// Tabulate coordinate map basis functions
	xt::xtensor<double, 4> table = cmap.tabulate(1, points);
	xt::xtensor<double, 2> phi = xt::view(table, 0, xt::all(), xt::all(), 0);
	xt::xtensor<double, 3> dphi = xt::view(table, xt::range(1, tdim+1), xt::all(), xt::all());

	// Create placeholder for Jacobian data
	xt::xtensor<double, 4> J_inv({ncells, nq, tdim, gdim});
	xt::xtensor<double, 2> J({tdim, gdim});
	xt::xtensor<double, 2> detJ({ncells, nq});
	xt::xtensor<double, 2> coords({num_nodes, gdim});

	// Compute Jacobian data
	tcb::span<const int> x_dofs;
	for (std::size_t c = 0; c < ncells; c++){
		// Get cell coordinates/geometry
		x_dofs = x_dofmap.links(c);

		// Copying x to coords
		for (std::size_t i = 0; i < x_dofs.size(); i++){
			std::copy_n(xt::row(x, x_dofs[i]).begin(), 3, xt::row(coords, i).begin());
		}

		// Computing determinant and inverse of the Jacobian
		J.fill(0.0);
		for (std::size_t q = 0; q < nq; q++){
			xt::view(J, 0, 0) = xt::sum(xt::view(coords, xt::all(), 0) * xt::view(dphi, 0, q, xt::all()));
			xt::view(J, 0, 1) = xt::sum(xt::view(coords, xt::all(), 1) * xt::view(dphi, 0, q, xt::all()));
			xt::view(J, 1, 0) = xt::sum(xt::view(coords, xt::all(), 0) * xt::view(dphi, 1, q, xt::all()));
			xt::view(J, 1, 1) = xt::sum(xt::view(coords, xt::all(), 1) * xt::view(dphi, 1, q, xt::all()));
			detJ(c, q) = xt::linalg::det(J) * weights[q];
			xt::view(J_inv, c, q, xt::all(), xt::all()) = xt::linalg::inv(J);
			std::cout << q << " " << xt::view(J_inv, c, q, xt::all(), xt::all()) << std::endl;
		}
	}
	return {J, detJ};
}