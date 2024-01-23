#pragma once

#include "forms.h"

#include <fstream>
#include <memory>
#include <string>

#include <dolfinx.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/Vector.h>

using namespace dolfinx;

namespace kernels {
// Copy data from a la::Vector in to a la::Vector out, including ghost entries.
template <typename T>
void copy(const la::Vector<T>& in, la::Vector<T>& out) {
  std::span<const T> _in = in.array();
  std::span<T> _out = out.mutable_array();
  std::copy(_in.begin(), _in.end(), _out.begin());
}

/// Compute vector r = alpha*x + y
/// @param r Result
/// @param alpha
/// @param x
/// @param y
template <typename T>
void axpy(la::Vector<T>& r, T alpha, const la::Vector<T>& x,
          const la::Vector<T>& y) {
  std::transform(x.array().begin(), x.array().begin() + x.map()->size_local(), y.array().begin(),
                 r.mutable_array().begin(),
                 [&alpha](const T& vx, const T& vy) { return vx * alpha + vy; });
}

} // namespace kernels

/// Solver for the Westervelt equation.
/// This solver uses GLL lattice and GLL quadrature such that it produces
/// a diagonal mass matrix.
/// @param [in] Mesh The mesh
/// @param [in] FacetTags The boundary facet tags
/// @param [in] speedOfSound A DG function defining the speed of sound within the domain
/// @param [in] density A DG function defining the densities within the domain
/// @param [in] diffusivityOfSound A DG function defining the diffusivity of sound within the domain
/// @param [in] coefficientOfNonlinearity A DG function defining the nonlinearity coefficient within the domain
/// @param [in] sourceFrequency The source frequency
/// @param [in] sourceAmplitude The source amplitude
/// @param [in] sourceSpeed The medium speed of sound that is in contact with the source
template <typename T, int P>
class WesterveltSpectral {
public:
  LossySpectral(
    std::shared_ptr<mesh::Mesh> Mesh,
    std::shared_ptr<mesh::MeshTags<std::int32_t>> FacetTags,
    std::shared_ptr<fem::Function<T>> speedOfSound,
    std::shared_ptr<fem::Function<T>> density,
    std::shared_ptr<fem::Function<T>> diffusivityOfSound,
    std::shared_ptr<fem::Function<T>> coefficientOfNonlinearity,
    const T& sourceFrequency, const T& sourceAmplitude,
    const T& sourceSpeed)
  {
    // MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Physical parameters
    c0 = speedOfSound;
    rho0 = density;
    delta0 = diffusivityOfSound;
    beta0 = coefficientOfNonlinearity;
    freq = sourceFrequency;
    w0 = 2 * M_PI * sourceFrequency;
    p0 = sourceAmplitude;
    s0 = sourceSpeed;
    period = 1.0 / sourceFrequency;
    window_length = 4.0;

    // Mesh data
    mesh = Mesh;
    ft = FacetTags;


  }
}