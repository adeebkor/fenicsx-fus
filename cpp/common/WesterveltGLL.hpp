#include "forms.h"

#include <dolfinx.h>
#include <dolfinx/la/Vector.h>
#include <memory>

using namespace dolfinx;

namespace kernels {
// Copy data from a la::Vector in to a la::Vector out, including ghost entries.
void copy(const la::Vector<double>& in, la::Vector<double>& out){
  xtl::span<const double> _in = in.array();
  xtl::span<double> _out = out.mutable_array();
  std::copy(_in.cbegin(), _in.cend(), _out.begin());
}

/// Compute vector r = alpha*x + y
/// @param r Result
/// @param alpha
/// @param x
/// @param y
void axpy(la::Vector<double>& r, double alpha, const la::Vector<double>& x,
          const la::Vector<double>& y) {
  std::transform(x.array().cbegin(), x.array().cbegin() + x.map()->size_local(), y.array().cbegin(),
                 r.mutable_array().begin(),
                 [&alpha](const double& vx, const double& vy) { return vx * alpha + vy; });
}

} // namespace kernels

class WesterveltGLL {
private:
  int rank;      // MPI rank
protected:
  int k_;        // degree of basis function
  double c0_;    // speed of sound (m/s)
  double freq0_; // source frequency (Hz)
  double p0_;    // pressure amplitude (Pa)
  double w0_;    // angular frequency (rad/s)
  double delta_; // diffusivity of sound
  double beta_;  // coefficient of nonlinearity
  double rho0_;  // density of medium (kg/m^3)
  double T_;     // period (s)
  double alpha_;
  double window_, dwindow_;

  std::shared_ptr<fem::Constant<double>> c0, delta, beta, rho0;
  std::shared_ptr<fem::Form<double>> a, L;
  std::shared_ptr<fem::Function<double>> u, v, g, dg, u_n, v_n;
  std::shared_ptr<la::Vector<double>> m, b;

  xtl::span<double> _g, _dg, out;
  xtl::span<const double> m_, b_;
  tcb::span<double> _m, _b;

  std::shared_ptr<const common::IndexMap> index_map;
  int bs;

public:
  std::shared_ptr<fem::FunctionSpace> V;

  WesterveltGLL(std::shared_ptr<mesh::Mesh> Mesh,
                std::shared_ptr<mesh::MeshTags<std::int32_t>> Meshtags, int& degreeOfBasis,
                double& speedOfSound, double& sourceFrequency, double& pressureAmplitude,
                double& diffusivityOfSound, double& coeffOfNonlinearity, double& densityOfMedium) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_forms_a, "u", Mesh));

    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    c0 = std::make_shared<fem::Constant<double>>(speedOfSound);
    delta = std::make_shared<fem::Constant<double>>(diffusivityOfSound);
    beta = std::make_shared<fem::Constant<double>>(coeffOfNonlinearity);
    rho0 = std::make_shared<fem::Constant<double>>(densityOfMedium);

    u = std::make_shared<fem::Function<double>>(V);
    v = std::make_shared<fem::Function<double>>(V);
    g = std::make_shared<fem::Function<double>>(V);
    dg = std::make_shared<fem::Function<double>>(V);
    u_n = std::make_shared<fem::Function<double>>(V);
    v_n = std::make_shared<fem::Function<double>>(V);

    _g = g->x()->mutable_array();
    _dg = dg->x()->mutable_array();

    // Physical parameters
    k_ = degreeOfBasis;
    c0_ = speedOfSound;
    freq0_ = sourceFrequency;
    p0_ = pressureAmplitude;
    w0_ = 2.0 * M_PI * freq0_;
    delta_ = diffusivityOfSound;
    beta_ = coeffOfNonlinearity;
    rho0_ = densityOfMedium;
    T_ = 1 / freq0_;
    alpha_ = 4.0;

    // Create LHS form
    xtl::span<double> _u = u->x()->mutable_array();
    std::fill(_u.begin(), _u.end(), 1.0);

    a = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_forms_a, {V},
                                 {{"u", u}, {"u_n", u_n}},
                                 {{"c0", c0}, {"rho0", rho0}, {"beta", beta}, {"delta", delta}},
                                 {{dolfinx::fem::IntegralType::exterior_facet, &(*Meshtags)}}));

    // Allocate memory for the LHS
    m = std::make_shared<la::Vector<double>>(index_map, bs);
    _m = m->mutable_array();

    // Create RHS form
    L = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_forms_L, {V}, 
                                 {{"u_n", u_n}, {"g", g}, {"v_n", v_n}, {"dg", dg}},
                                 {{"c0", c0}, {"rho0", rho0}, {"beta", beta}, {"delta", delta}},
                                 {{dolfinx::fem::IntegralType::exterior_facet, &(*Meshtags)}}));

    // Allocate memory for the RHS
    b = std::make_shared<la::Vector<double>>(index_map, bs);
    _b = b->mutable_array();
  }

  /// Set the initial values of u and v, i.e. u_0 and v_0
  void init() {
    tcb::span<double> u_0 = u_n->x()->mutable_array();
    tcb::span<double> v_0 = v_n->x()->mutable_array();

    std::fill(u_0.begin(), u_0.end(), 0.0);
    std::fill(v_0.begin(), v_0.end(), 0.0);
  }

  /// Evaluate du/dt = f0(t, u, v)
  /// @param t Current time, i.e. tn
  /// @param u Current u, i.e. un
  /// @param v Current v, i.e. vn
  /// @param result Result, i.e. dun/dtn
  void f0(double& t, std::shared_ptr<la::Vector<double>> u,
          std::shared_ptr<la::Vector<double>> v, std::shared_ptr<la::Vector<double>> result) {
    kernels::copy(*v, *result);
  }

  /// Evaluate dv/dt = f1(t, u, v)
  /// @param t Current time, i.e. tn
  /// @param u Current u, i.e. un
  /// @param v Current v, i.e. vn
  /// @param result Result, i.e. dvn/dtn
  void f1(double& t, std::shared_ptr<la::Vector<double>> u,
          std::shared_ptr<la::Vector<double>> v, std::shared_ptr<la::Vector<double>> result) {
  
    // Apply windowing
    if (t < T_ * alpha_) {
        window_ = 0.5 * (1.0 - cos(freq0_ * M_PI * t / alpha_));
        dwindow_ = 0.5 * M_PI * freq0_ / alpha_ * sin(freq0_ * M_PI * t / alpha_);
    } else {
        window_ = 1.0;
        dwindow_ = 0.0;
    }

    // Update boundary condition
    std::fill(_g.begin(), _g.end(), window_ * p0_ * w0_ / c0_ * cos(w0_ * t));
    std::fill(_dg.begin(), _dg.end(), dwindow_ * p0_ * w0_ / c0_ * cos(w0_ * t)
                                      - window_ * p0_ * w0_ * w0_ / c0_ * sin(w0_ * t));

    u->scatter_fwd();
    kernels::copy(*u, *u_n->x());

    v->scatter_fwd();
    kernels::copy(*v, *v_n->x());

    // TODO: Compute coefficients

    // Assemble LHS
    std::fill(_m.begin(), _m.end(), 0.0);
    fem::assemble_vector(_m, *a);
    m->scatter_rev(common::IndexMap::Mode::add);

    // Assemble RHS
    std::fill(_b.begin(), _b.end(), 0.0);
    fem::assemble_vector(_b, *L);
    b->scatter_rev(common::IndexMap::Mode::add);

    // Solve
    {
      out = result->mutable_array();
      b_ = b->array();
      m_ = m->array();

      // Element wise division
      // out[i] = b[i]/m[i]
      std::transform(b_.begin(), b_.end(), m_.begin(), out.begin(),
                     [](const double& bi, const double& mi) { return bi/mi; });

    }
  }

  void rk4(double& startTime, double& finalTime, double& timeStep) {

    double t = startTime;
    double tf = finalTime;
    double dt = timeStep;
    int step = 0;
    int nstep = (finalTime - startTime) / timeStep + 1;

    std::shared_ptr<la::Vector<double>> u_, v_, un, vn, u0, v0, ku, kv;

    // Placeholder vectors at time step n
    u_ = std::make_shared<la::Vector<double>>(index_map, bs);
    v_ = std::make_shared<la::Vector<double>>(index_map, bs);
    
    kernels::copy(*u_n->x(), *u_);
    kernels::copy(*v_n->x(), *v_);

    // Placeholder vectors at intermediate time step n
    un = std::make_shared<la::Vector<double>>(index_map, bs);
    vn = std::make_shared<la::Vector<double>>(index_map, bs);

    // Placeholder vectors at start of time step
    u0 = std::make_shared<la::Vector<double>>(index_map, bs);
    v0 = std::make_shared<la::Vector<double>>(index_map, bs);

    // Placeholder at k intermediate time step
    ku = std::make_shared<la::Vector<double>>(index_map, bs);
    kv = std::make_shared<la::Vector<double>>(index_map, bs);

    kernels::copy(*u_, *ku);
    kernels::copy(*v_, *kv);

    // Runge-Kutta timestepping data
    int n_RK = 4;
    xt::xarray<double> a_runge{0.0, 0.5, 0.5, 1.0};
    xt::xarray<double> b_runge{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    xt::xarray<double> c_runge{0.0, 0.5, 0.5, 1.0};

    // RK variables
    double tn;

    // Write to VTX
    dolfinx::io::VTXWriter file(MPI_COMM_WORLD, "u.pvd", {u_n});
    file.write(t);

    while (t < tf) {
      dt = std::min(dt, tf - t);

      // Store solution at start of time step
      kernels::copy(*u_, *u0);
      kernels::copy(*v_, *v0);

      // Runge-Kutta step
      for (int i = 0; i < n_RK; i++) {
        kernels::copy(*u0, *un);
        kernels::copy(*v0, *vn);
      
        kernels::axpy(*un, dt*a_runge(i), *ku, *un);
        kernels::axpy(*vn, dt*a_runge(i), *kv, *vn);

        // RK time evaluation
        tn = t + c_runge(i) * dt;

        // Compute RHS vector
        f0(tn, un, vn, ku);
        f1(tn, un, vn, kv);

        // Update solution
        kernels::axpy(*u_, dt * b_runge(i), *ku, *u_);
        kernels::axpy(*v_, dt * b_runge(i), *kv, *v_);
      }

      // Update time
      t += dt;
      step += 1;

      if (step % 50 == 0){
        kernels::copy(*u_, *u_n->x());
        file.write(t);
        if (rank == 0){
          std::cout << "t: " << t << ",\t Steps: " << step << "/" << nstep << std::endl;
        }
      }
    }

    if (rank == 0){
      std::cout << "t: " << t << ",\t Steps: " << step << "/" << nstep << std::endl;
    }

    // Prepare solution at final time
    kernels::copy(*u_, *u_n->x());
    kernels::copy(*v_, *v_n->x());
    u_n->x()->scatter_fwd();
    v_n->x()->scatter_fwd();

    file.write(t);
    file.close();
  }
};
