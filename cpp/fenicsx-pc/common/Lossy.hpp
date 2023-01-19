#pragma once

#include "forms.h"
#include "precompute_op.hpp"

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

/// Solver for the 2D second order linear wave equation with attenuation.
/// This solver uses GLL lattice and GLL quadrature such that it produces
/// a diagonal mass matrix.
/// @param [in] Mesh The mesh
/// @param [in] FacetTags The boundary facet tags
/// @param [in] speedOfSound A DG function defining the speed of sound within the domain
/// @param [in] density A DG function defining the densities within the domain
/// @param [in] diffusivityOfSound A DG function defining the diffusivity of sound within the domain
/// @param [in] sourceFrequency The source frequency
/// @param [in] sourceAmplitude The source amplitude
/// @param [in] sourceSpeed The medium speed of sound that is in contact with the source
template <typename T, int P>
class LossySpectral2D {
public:
  LossySpectral2D(
    std::shared_ptr<mesh::Mesh> Mesh,
    std::shared_ptr<mesh::MeshTags<std::int32_t>> FacetTags,
    std::shared_ptr<fem::Function<T>> speedOfSound,
    std::shared_ptr<fem::Function<T>> density,
    std::shared_ptr<fem::Function<T>> diffusivityOfSound,
    const T& sourceFrequency, const T& sourceAmplitude)
  {
    // MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Physical parameters
    c0 = speedOfSound;
    rho0 = density;
    delta0 = diffusivityOfSound;
    freq = sourceFrequency;
    w0 = 2 * M_PI * sourceFrequency;
    p0 = sourceAmplitude;
    s0 = sourceSpeed;
    period = 1.0 / sourceFrequency;
    window_length = 4.0;

    // Mesh data
    mesh = Mesh;
    ft = FacetTags;

    // Define function space
    V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_a, "u", mesh));
    
    // Define field functions
    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    // Define source function
    g = std::make_shared<fem::Function<T>>(V);
    g_ = g->x()->mutable_array();
    dg = std::make_shared<fem::Function<T>>(V);
    dg_ = dg->x()->mutable_array();
    
    // Define forms
    std::span<T> u_ = u->x()->mutable_array();
    std::fill(u_.begin(), u_.end(), 1.0);

    // Define LHS form
    a = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_forms_a, {V}, 
        {{"u", u}, {"c0", c0}, {"rho0", rho0}, {"delta0", delta0}},
        {},
        {{dolfinx::fem::IntegralType::exterior_facet, &(*ft)}}));
    
    m = std::make_shared<la::Vector<T>>(index_map, bs);
    m_ = m->mutable_array();
    std::fill(m_.begin(), m_.end(), 0.0);
    fem::assemble_vector(m_, *a);
    m->scatter_rev(std::plus<T>());

    // Define RHS form
    L = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_L, {V}, 
                          {{"g", g}, {"dg", dg}, {"v_n", v_n}, 
                           {"c0", c0}, {"rho0", rho0}, {"delta0", delta0}},
                          {}, 
                          {{dolfinx::fem::IntegralType::exterior_facet,
                            &(*ft)}}));
    b = std::make_shared<la::Vector<T>>(index_map, bs);
    b_ = b->mutable_array();

    // Define operators
    lin_op = std::make_shared<StiffnessSpectral2D<T, P>>(V);
    att_op = std::make_shared<StiffnessSpectral2D<T, P>>(V);

    // Define coefficient for the operators
    std::span<const T> c0_ = c0->x()->array();
    std::span<const T> rho0_ = rho0->x()->array();
    std::span<const T> delta0_ = delta0->x()->array();

    lin_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    lin_coeff_ = lin_coeff->x()->mutable_array();

    att_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    att_coeff_ = att_coeff


  }
}