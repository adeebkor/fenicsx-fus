#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>

int main(int argc, char* argv[])
{
    xt::xarray<double> arr1 {1.0, 2.0, 3.0};
    xt::xarray<double> arr2 {9.0, 8.0, 7.5};

    auto f = 2 * arr1 + arr2;

    double a = f(0);
    double b = f(2);

    std::cout << a << " " << b << std::endl;

    return 0;
}