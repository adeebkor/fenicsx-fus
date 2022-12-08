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

/// Solver for the second order linear wave equation.
/// This solver uses GLL lattice and GLL quadrature such that it produces
/// a diagonal mass matrix.
/// @param [in] Mesh The mesh
/// @param [in] FacetTags The boundary facet tags
/// @param [in] speedOfSound A DG function defining the speed of sound within the domain
/// @param [in] density A DG function defining the densities within the domain
/// @param [in] sourceFrequency The source frequency
/// @param [in] sourceAmplitude The source amplitude
/// @param [in] sourceSpeed The medium speed of sound that is in contact with the source
template <typename T, int P>
class LinearSpectral{
public:
  LinearSpectral(
    std::shared_ptr<mesh::Mesh> Mesh,
    std::shared_ptr<mesh::MeshTags<std::int32_t>> FacetTags,
    std::shared_ptr<fem::Function<T>> speedOfSound,
    std::shared_ptr<fem::Function<T>> density,
    const T& sourceFrequency, const T& sourceAmplitude,
    const T& sourceSpeed)
  {
    // MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Physical parameters
    c0 = speedOfSound;
    rho0 = density;
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

    u = std::make_shared<fem::Function<T>>(V);
    u_n = std::make_shared<fem::Function<T>>(V);
    v_n = std::make_shared<fem::Function<T>>(V);

    // Define source function
    g = std::make_shared<fem::Function<T>>(V);
    g_ = g->x()->mutable_array();

    // Define forms
    std::span<T> u_ = u->x()->mutable_array();
    std::fill(u_.begin(), u_.end(), 1.0);

    // Define LHS form
    a = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_a, {V},
                          {{"u", u}, {"c0", c0}, {"rho0", rho0}}, {}, {}));

    m = std::make_shared<la::Vector<T>>(index_map, bs);
    m_ = m->mutable_array();
    std::fill(m_.begin(), m_.end(), 0.0);
    fem::assemble_vector(m_, *a);
    m->scatter_rev(std::plus<T>());

    // Define RHS form
    L = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_L, {V}, 
                          {{"g", g}, {"u_n", u_n}, {"v_n", v_n}, 
                           {"c0", c0}, {"rho0", rho0}},
                          {}, 
                          {{dolfinx::fem::IntegralType::exterior_facet,
                            &(*ft)}}));
    b = std::make_shared<la::Vector<T>>(index_map, bs);
    b_ = b->mutable_array();
  }

  /// Set the initial values of u and v, i.e. u_0 and v_0
  void init() {
    u_n->x()->set(0.0);
    v_n->x()->set(0.0);
  }

  /// Evaluate du/dt = f0(t, u, v)
  /// @param[in] t Current time, i.e. tn
  /// @param[in] u Current u, i.e. un
  /// @param[in] v Current v, i.e. vn
  /// @param[out] result Result, i.e. dun/dtn
  void f0(T& t, std::shared_ptr<la::Vector<T>> u, std::shared_ptr<la::Vector<T>> v,
          std::shared_ptr<la::Vector<T>> result) {
    kernels::copy<T>(*v, *result);
  }

  /// Evaluate dv/dt = f1(t, u, v)
  /// @param[in] t Current time, i.e. tn
  /// @param[in] u Current u, i.e. un
  /// @param[in] v Current v, i.e. vn
  /// @param[out] result Result, i.e. dvn/dtn
  void f1(T& t, std::shared_ptr<la::Vector<T>> u, std::shared_ptr<la::Vector<T>> v,
          std::shared_ptr<la::Vector<T>> result) {

    // Apply windowing
    if (t < period * window_length) {
      window = 0.5 * (1.0 - cos(freq * M_PI * t / window_length));
    } else {
      window = 1.0;
    }

    // Update boundary condition
    std::fill(g_.begin(), g_.end(), window * p0 * w0 / s0 * cos(w0 * t));

    u->scatter_fwd();
    kernels::copy<T>(*u, *u_n->x());

    v->scatter_fwd();
    kernels::copy<T>(*v, *v_n->x());

    // Assemble RHS
    std::fill(b_.begin(), b_.end(), 0.0);
    fem::assemble_vector(b_, *L);
    b->scatter_rev(std::plus<T>());

    // Solve
    // TODO: Divide is more expensive than multiply.
    // We should store the result of 1/m in a vector and apply and element wise vector
    // multiplication, since m doesn't change for linear wave propagation.
    {
      out = result->mutable_array();
      _b = b->array();
      _m = m->array();

      // Element wise division
      // out[i] = b[i]/m[i]
      std::transform(_b.begin(), _b.end(), _m.begin(), out.begin(),
                     [](const T& bi, const T& mi) { return bi / mi; });
    }
  }

  /// Runge-Kutta 4th order solver
  /// @param[in] startTime initial time of the solver
  /// @param[in] finalTime final time of the solver
  /// @param[in] timeStep  time step size of the solver
  void rk4(const T& startTime, const T& finalTime, const T& timeStep) {

    // Time-stepping parameters
    T t = startTime;
    T tf = finalTime;
    T dt = timeStep;
    int totalStep = (finalTime - startTime) / timeStep + 1;
    int step = 0;

    // Time-stepping vectors
    std::shared_ptr<la::Vector<T>> u_, v_, un, vn, u0, v0, ku, kv;

    // Placeholder vectors at time step n
    u_ = std::make_shared<la::Vector<T>>(index_map, bs);
    v_ = std::make_shared<la::Vector<T>>(index_map, bs);

    kernels::copy<T>(*u_n->x(), *u_);
    kernels::copy<T>(*v_n->x(), *v_);

    // Placeholder vectors at intermediate time step n
    un = std::make_shared<la::Vector<T>>(index_map, bs);
    vn = std::make_shared<la::Vector<T>>(index_map, bs);

    // Placeholder vectors at start of time step
    u0 = std::make_shared<la::Vector<T>>(index_map, bs);
    v0 = std::make_shared<la::Vector<T>>(index_map, bs);

    // Placeholder at k intermediate time step
    ku = std::make_shared<la::Vector<T>>(index_map, bs);
    kv = std::make_shared<la::Vector<T>>(index_map, bs);

    kernels::copy<T>(*u_, *ku);
    kernels::copy<T>(*v_, *kv);

    // Runge-Kutta 4th order time-stepping data
    std::array<T, 4> a_runge = {0.0, 0.5, 0.5, 1.0};
    std::array<T, 4> b_runge = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    std::array<T, 4> c_runge = {0.0, 0.5, 0.5, 1.0};

    // RK variables
    T tn;

    while (t < tf) {
      dt = std::min(dt, tf - t);

      // Store solution at start of time step
      kernels::copy<T>(*u_, *u0);
      kernels::copy<T>(*v_, *v0);

      // Runge-Kutta 4th order step
      for (int i = 0; i < 4; i++) {
        kernels::copy<T>(*u0, *un);
        kernels::copy<T>(*v0, *vn);

        kernels::axpy<T>(*un, dt * a_runge[i], *ku, *un);
        kernels::axpy<T>(*vn, dt * a_runge[i], *kv, *vn);

        // RK time evaluation
        tn = t + c_runge[i] * dt;

        // Compute RHS vector
        f0(tn, un, vn, ku);
        f1(tn, un, vn, kv);

        // Update solution
        kernels::axpy<T>(*u_, dt * b_runge[i], *ku, *u_);
        kernels::axpy<T>(*v_, dt * b_runge[i], *kv, *v_);
      }

      // Update time
      t += dt;
      step += 1;

      if (step % 100 == 0) {
        if (mpi_rank == 0) {
          std::cout << "t: " << t 
                    << ",\t Steps: " << step 
                    << "/" << totalStep << std::endl;
        }
      }
    }

    // Prepare solution at final time
    kernels::copy<T>(*u_, *u_n->x());
    kernels::copy<T>(*v_, *v_n->x());
    u_n->x()->scatter_fwd();
    v_n->x()->scatter_fwd();

  }

  std::shared_ptr<fem::Function<T>> u_sol() const {
    return u_n;
  }

private:
  int mpi_rank, mpi_size;  // MPI rank and size
  int bs;  // block size
  T freq;  // source frequency (Hz)
  T p0;  // source amplitude (Pa)
  T w0;  // angular frequency  (rad/s)
  T s0;  // speed (m/s)
  T period, window_length, window;

  std::shared_ptr<mesh::Mesh> mesh;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> ft;
  std::shared_ptr<const common::IndexMap> index_map;
  std::shared_ptr<fem::FunctionSpace> V;
  std::shared_ptr<fem::Function<T>> u, u_n, v_n, g, c0, rho0;
  std::shared_ptr<fem::Form<T>> a, L;
  std::shared_ptr<la::Vector<T>> m, b;

  std::span<T> g_, m_, b_, out;
  std::span<const T> _m, _b;
};

// Note:
// mutable array -> variable_name_
// array -> _variable_name