#include <istream>
#include <fstream>
#include <iostream>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/math.h>
#include <xtensor/xio.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xnpy.hpp>

void tabulate_basis_and_permutation(int p=4, int q=6){
    // Tabulate quadrature points and weights
    auto family = basix::element::family::P;
    auto cell_type = basix::cell::type::hexahedron;
    auto quad_scheme = basix::quadrature::type::gll;
    auto [points, weights] = basix::quadrature::make_quadrature(quad_scheme, cell_type, q);
    auto variant = basix::element::lagrange_variant::gll_warped;
    auto element = basix::create_element(family, cell_type, p, variant);
    xt::xtensor<double, 4> basis = element.tabulate(1, points);

    xt::xtensor<double, 2> basis0 = xt::view(basis, 0, xt::all(), xt::all(), 0);
    auto idx0 = xt::argwhere(xt::isclose(basis0, 0.0));
    xt::index_view(basis0, idx0) = 0.0;
    auto idx1 = xt::argwhere(xt::isclose(basis0, 1.0));
    xt::index_view(basis0, idx1) = 1.0;

    auto idx = xt::from_indices(idx1);

    std::ofstream out_file;
    out_file.open("out.csv");

    xt::dump_csv(out_file, basis0);

    std::cout << "permutation: \n" << idx << std::endl;
}

int main(){

    tabulate_basis_and_permutation();

    return 0;
}

