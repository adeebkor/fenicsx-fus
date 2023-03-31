#pragma once

#include "forms.h"
#include "spectral_op.hpp"

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


/// Solver for the 2D second order Westervelt equation.
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
class WesterveltSpectral2D {
public:
  WesterveltSpectral2D(
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

    // Define function space
    V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_a, "u", mesh));
    
    // Define field functions
    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    u = std::make_shared<fem::Function<T>>(V);
    u_n = std::make_shared<fem::Function<T>>(V);
    v_n = std::make_shared<fem::Function<T>>(V);
    w_n = std::make_shared<fem::Function<T>>(V);

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
    
    m0 = std::make_shared<la::Vector<T>>(index_map, bs);
    m0_ = m0->mutable_array();
    std::fill(m0_.begin(), m0_.end(), 0.0);
    fem::assemble_vector(m0_, *a);

    m = std::make_shared<la::Vector<T>>(index_map, bs);
    m_ = m->mutable_array();

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

    nlin1_op = std::make_shared<MassSpectral2D<T, P>>(V);
    nlin2_op = std::make_shared<MassSpectral2D<T, P>>(V);

    // Define coefficient for the operators
    std::span<const T> c0_ = c0->x()->array();
    std::span<const T> rho0_ = rho0->x()->array();
    std::span<const T> delta0_ = delta0->x()->array();
    std::span<const T> beta0_ = beta0->x()->array();

    lin_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    lin_coeff_ = lin_coeff->x()->mutable_array();

    att_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    att_coeff_ = att_coeff->x()->mutable_array();

    nlin1_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    nlin1_coeff_ = nlin1_coeff->x()->mutable_array();

    nlin2_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    nlin2_coeff_ = nlin2_coeff->x()->mutable_array();

    for (std::size_t i = 0; i < rho0_.size(); ++i) {
      lin_coeff_[i] = - 1.0 / rho0_[i];
      att_coeff_[i] = - delta0_[i] / rho0_[i] / c0_[i] / c0_[i];
      nlin1_coeff_[i] = - 2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] 
        / c0_[i] / c0_[i] / c0_[i];
      nlin2_coeff_[i] = 2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] 
        / c0_[i] / c0_[i] / c0_[i];
    }

    lin_coeff->x()->scatter_fwd();
    att_coeff->x()->scatter_fwd();
    nlin1_coeff->x()->scatter_fwd();
    nlin2_coeff->x()->scatter_fwd();
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
      dwindow = 0.5 * M_PI * freq / window_length * sin(freq * M_PI * t 
                                                         / window_length);
    } else {
      window = 1.0;
      dwindow = 0.0;
    }

    // Update boundary condition (homogenous domain)
    std::fill(g_.begin(), g_.end(), window * p0 * w0 / s0 * cos(w0 * t));
    std::fill(dg_.begin(), dg_.end(), 
              dwindow * p0 * w0 / s0 * cos(w0 * t) 
                - window * p0 * w0 * w0 / s0 * sin(w0 * t));

    // Update boundary condition (heterogenous domain)
    // std::fill(g_.begin(), g_.end(), window * 2.0 * p0 * w0 / s0 * cos(w0 * t));
    // std::fill(dg_.begin(), dg_.end(), 
    //           dwindow * 2.0 * p0 * w0 / s0 * cos(w0 * t) 
    //             - window * 2.0 * p0 * w0 * w0 / s0 * sin(w0 * t));

    // Update fields
    u->scatter_fwd();
    kernels::copy<T>(*u, *u_n->x());

    v->scatter_fwd();
    kernels::copy<T>(*v, *v_n->x());

    std::transform(v_n->x()->array().begin(), v_n->x()->array().end(),
                   w_n->x()->mutable_array().begin(),
                   [&](const T& vx) { return vx * vx; });

    // Assemble LHS
    std::fill(m_.begin(), m_.end(), 0.0);
    nlin1_op->operator()(*u_n->x(), nlin1_coeff_, *m);
    std::transform(m0->array().begin(), m0->array().end(), m->array().begin(),
                   m->mutable_array().begin(),
                   [&](const T& x, const T& y) { return x + y; }); 
    m->scatter_rev(std::plus<T>());

    // Assemble RHS
    std::fill(b_.begin(), b_.end(), 0.0);
    lin_op->operator()(*u_n->x(), lin_coeff_, *b);
    att_op->operator()(*v_n->x(), att_coeff_, *b);
    nlin2_op->operator()(*w_n->x(), nlin2_coeff_, *b);
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
                    << "/" << totalStep
                    << "\t" << u_->array()[0] << std::endl;
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

  std::int64_t number_of_dofs() const {
    return V->dofmap()->index_map->size_global();
  }

private:
  int mpi_rank, mpi_size;  // MPI rank and size
  int bs;  // block size
  T freq;  // source frequency (Hz)
  T p0;  // source amplitude (Pa)
  T w0;  // angular frequency  (rad/s)
  T s0;  // speed (m/s)
  T period, window_length, window, dwindow;

  std::shared_ptr<mesh::Mesh> mesh;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> ft;
  std::shared_ptr<const common::IndexMap> index_map;
  std::shared_ptr<fem::FunctionSpace> V;
  std::shared_ptr<fem::Function<T>> u, u_n, v_n, w_n, g, dg, c0, rho0, delta0, beta0;
  std::shared_ptr<fem::Form<T>> a, L;
  std::shared_ptr<la::Vector<T>> m, m0, b;

  std::span<T> g_, dg_, m_, m0_, b_, out;
  std::span<const T> _m, _b;

  // Operators
  std::shared_ptr<StiffnessSpectral2D<T, P>> lin_op, att_op;
  std::shared_ptr<MassSpectral2D<T, P>> nlin1_op, nlin2_op;

  // Operators' coefficients
  std::shared_ptr<fem::Function<T>> lin_coeff, att_coeff, nlin1_coeff, nlin2_coeff;
  std::span<T> lin_coeff_, att_coeff_, nlin1_coeff_, nlin2_coeff_;
};


/// Solver for the 3D second order Westervelt equation.
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
class WesterveltSpectral3D {
public:
  WesterveltSpectral3D(
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

    // Define function space
    V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_a, "u", mesh));
    
    // Define field functions
    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    u = std::make_shared<fem::Function<T>>(V);
    u_n = std::make_shared<fem::Function<T>>(V);
    v_n = std::make_shared<fem::Function<T>>(V);
    w_n = std::make_shared<fem::Function<T>>(V);

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
    
    m0 = std::make_shared<la::Vector<T>>(index_map, bs);
    m0_ = m0->mutable_array();
    std::fill(m0_.begin(), m0_.end(), 0.0);
    fem::assemble_vector(m0_, *a);

    m = std::make_shared<la::Vector<T>>(index_map, bs);
    m_ = m->mutable_array();

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
    lin_op = std::make_shared<StiffnessSpectral3D<T, P>>(V);
    att_op = std::make_shared<StiffnessSpectral3D<T, P>>(V);

    nlin1_op = std::make_shared<MassSpectral3D<T, P>>(V);
    nlin2_op = std::make_shared<MassSpectral3D<T, P>>(V);

    // Define coefficient for the operators
    std::span<const T> c0_ = c0->x()->array();
    std::span<const T> rho0_ = rho0->x()->array();
    std::span<const T> delta0_ = delta0->x()->array();
    std::span<const T> beta0_ = beta0->x()->array();

    lin_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    lin_coeff_ = lin_coeff->x()->mutable_array();

    att_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    att_coeff_ = att_coeff->x()->mutable_array();

    nlin1_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    nlin1_coeff_ = nlin1_coeff->x()->mutable_array();

    nlin2_coeff = std::make_shared<fem::Function<T>>(rho0->function_space());
    nlin2_coeff_ = nlin2_coeff->x()->mutable_array();

    for (std::size_t i = 0; i < rho0_.size(); ++i) {
      lin_coeff_[i] = - 1.0 / rho0_[i];
      att_coeff_[i] = - delta0_[i] / rho0_[i] / c0_[i] / c0_[i];
      nlin1_coeff_[i] = - 2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] 
        / c0_[i] / c0_[i] / c0_[i];
      nlin2_coeff_[i] = 2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] 
        / c0_[i] / c0_[i] / c0_[i];
    }

    lin_coeff->x()->scatter_fwd();
    att_coeff->x()->scatter_fwd();
    nlin1_coeff->x()->scatter_fwd();
    nlin2_coeff->x()->scatter_fwd();
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
      dwindow = 0.5 * M_PI * freq / window_length * sin(freq * M_PI * t 
                                                         / window_length);
    } else {
      window = 1.0;
      dwindow = 0.0;
    }

    // Update boundary condition (homogenous domain)
    // std::fill(g_.begin(), g_.end(), window * p0 * w0 / s0 * cos(w0 * t));
    // std::fill(dg_.begin(), dg_.end(), 
    //           dwindow * p0 * w0 / s0 * cos(w0 * t) 
    //             - window * p0 * w0 * w0 / s0 * sin(w0 * t));

    // Update boundary condition (heterogenous domain)
    std::fill(g_.begin(), g_.end(), window * 2.0 * p0 * w0 / s0 * cos(w0 * t));
    std::fill(dg_.begin(), dg_.end(), 
              dwindow * 2.0 * p0 * w0 / s0 * cos(w0 * t) 
                - window * 2.0 * p0 * w0 * w0 / s0 * sin(w0 * t));

    // Update fields
    u->scatter_fwd();
    kernels::copy<T>(*u, *u_n->x());

    v->scatter_fwd();
    kernels::copy<T>(*v, *v_n->x());

    std::transform(v_n->x()->array().begin(), v_n->x()->array().end(),
                   w_n->x()->mutable_array().begin(),
                   [&](const T& vx) { return vx * vx; });

    // Assemble LHS
    std::fill(m_.begin(), m_.end(), 0.0);
    nlin1_op->operator()(*u_n->x(), nlin1_coeff_, *m);
    std::transform(m0->array().begin(), m0->array().end(), m->array().begin(),
                   m->mutable_array().begin(),
                   [&](const T& x, const T& y) { return x + y; }); 
    m->scatter_rev(std::plus<T>());

    // Assemble RHS
    std::fill(b_.begin(), b_.end(), 0.0);
    lin_op->operator()(*u_n->x(), lin_coeff_, *b);
    att_op->operator()(*v_n->x(), att_coeff_, *b);
    nlin2_op->operator()(*w_n->x(), nlin2_coeff_, *b);
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
    // ------------------------------------------------------------------------
    // Computing function evaluation parameters

    std::string fname;

    // Grid parameters
    const std::size_t Nr = 179;
    const std::size_t Nz = 357;

    // Create evaluation point coordinates
    std::vector<double> point_coordinates(3 * Nr * Nz);
    for (std::size_t i = 0; i < Nz; ++i) {
      for (std::size_t j = 0; j < Nr; ++j) {
        point_coordinates[3*j + 3*i*Nr] = j * 0.04 / (Nr - 1) - 0.02;
        point_coordinates[3*j + 3*i*Nr + 1] = 0.0;
        point_coordinates[3*j + 3*i*Nr + 2] = i * 0.08 / (Nz - 1);
      }
    }

    // Compute evaluation parameters
    auto bb_tree = geometry::BoundingBoxTree(*mesh, mesh->topology().dim());
    auto cell_candidates = compute_collisions(bb_tree, point_coordinates);
    auto colliding_cells = geometry::compute_colliding_cells(
      *mesh, cell_candidates, point_coordinates);

    std::vector<std::int32_t> cells;
    std::vector<double> points_on_proc;

    for (std::size_t i = 0; i < Nr*Nz; ++i) {
      auto link = colliding_cells.links(i);
      if (link.size() > 0) {
        points_on_proc.push_back(point_coordinates[3*i]);
        points_on_proc.push_back(point_coordinates[3*i + 1]);
        points_on_proc.push_back(point_coordinates[3*i + 2]);
        cells.push_back(link[0]);
      }
    }

    std::size_t num_points_local = points_on_proc.size() / 3;
    std::vector<T> u_eval(num_points_local);

    T* u_value = u_eval.data();
    double* p_value = points_on_proc.data();

    int numStepPerPeriod = period / timeStep + 3;
    int step_period = 0;
    // ------------------------------------------------------------------------

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
                    << "/" << totalStep
                    << "\t" << u_->array()[0] << std::endl;
        }
      }
      // ----------------------------------------------------------------------
      // Collect data
      if (t > 0.08 / s0 + 6.0 / freq && step_period < numStepPerPeriod) {
        kernels::copy(*u_, *u_n->x());
        u_n->x()->scatter_fwd();

        // Evaluate function
        u_n->eval(points_on_proc, {num_points_local, 3}, cells, u_eval,
                  {num_points_local, 1});
        u_value = u_eval.data();

        // Write evaluation from each process to a single text file
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < mpi_size; ++i) {
          if (mpi_rank == i) {
            fname = "/home/mabm4/rds/hpc-work/data/pressure_field_" + 
                    std::to_string(step_period) + ".txt";
            std::ofstream txt_file(fname, std::ios_base::app);
            for (std::size_t i = 0; i < num_points_local; ++i) {
              txt_file << *(p_value + 3 * i) << ","
                       << *(p_value + 3 * i + 2) << "," 
                       << *(u_value + i) << std::endl;
            }
            txt_file.close();
          }
          MPI_Barrier(MPI_COMM_WORLD);
        }
        step_period++;
      }
      // ----------------------------------------------------------------------
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

  std::int64_t number_of_dofs() const {
    return V->dofmap()->index_map->size_global();
  }

private:
  int mpi_rank, mpi_size;  // MPI rank and size
  int bs;  // block size
  T freq;  // source frequency (Hz)
  T p0;  // source amplitude (Pa)
  T w0;  // angular frequency  (rad/s)
  T s0;  // speed (m/s)
  T period, window_length, window, dwindow;

  std::shared_ptr<mesh::Mesh> mesh;
  std::shared_ptr<mesh::MeshTags<std::int32_t>> ft;
  std::shared_ptr<const common::IndexMap> index_map;
  std::shared_ptr<fem::FunctionSpace> V;
  std::shared_ptr<fem::Function<T>> u, u_n, v_n, w_n, g, dg, c0, rho0, delta0, beta0;
  std::shared_ptr<fem::Form<T>> a, L;
  std::shared_ptr<la::Vector<T>> m, m0, b;

  std::span<T> g_, dg_, m_, m0_, b_, out;
  std::span<const T> _m, _b;

  // Operators
  std::shared_ptr<StiffnessSpectral3D<T, P>> lin_op, att_op;
  std::shared_ptr<MassSpectral3D<T, P>> nlin1_op, nlin2_op;

  // Operators' coefficients
  std::shared_ptr<fem::Function<T>> lin_coeff, att_coeff, nlin1_coeff, nlin2_coeff;
  std::span<T> lin_coeff_, att_coeff_, nlin1_coeff_, nlin2_coeff_;
};

template <typename T>
const T compute_diffusivity_of_sound(const T w0, const T c0, const T alpha){
  const T diffusivity = 2*alpha*c0*c0*c0/w0/w0;

  return diffusivity;
}