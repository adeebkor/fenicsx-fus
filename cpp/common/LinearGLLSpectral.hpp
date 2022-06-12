#pragma once

#include "forms.h"
#include "spectral_mass_3d.hpp"
#include "spectral_stiffness_3d.hpp"

#include <fstream>
#include <memory>
#include <string>

#include <dolfinx.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/la/Vector.h>

using namespace dolfinx;

namespace kernels {
// Copy data from a la::Vector in to a la::Vector out, including ghost entries.
void copy(const la::Vector<double>& in, la::Vector<double>& out) {
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

template <int P>
class LinearGLLSpectral {
public:
  LinearGLLSpectral(std::shared_ptr<mesh::Mesh> Mesh,
                    std::shared_ptr<mesh::MeshTags<std::int32_t>> Meshtags, double& speedOfSound,
                    double& sourceFrequency, double& pressureAmplitude) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mesh = Mesh;
    V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_forms_a, "u", Mesh));

    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    c0 = std::make_shared<fem::Constant<double>>(speedOfSound);
    u = std::make_shared<fem::Function<double>>(V);
    v = std::make_shared<fem::Function<double>>(V);
    g = std::make_shared<fem::Function<double>>(V);
    u_n = std::make_shared<fem::Function<double>>(V);
    v_n = std::make_shared<fem::Function<double>>(V);

    _g = g->x()->mutable_array();

    // Physical parameters
    c0_ = speedOfSound;
    freq0_ = sourceFrequency;
    p0_ = pressureAmplitude;
    w0_ = 2.0 * M_PI * freq0_;
    T_ = 1.0 / freq0_;
    alpha_ = 4.0;

    // Create LHS form
    xtl::span<double> _u = u->x()->mutable_array();
    std::fill(_u.begin(), _u.end(), 1.0);

    a = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_forms_a, {V}, {{"u", u}}, {}, {}));

    m = std::make_shared<la::Vector<double>>(index_map, bs);
    _m = m->mutable_array();
    std::fill(_m.begin(), _m.end(), 0.0);
    fem::assemble_vector(_m, *a);
    m->scatter_rev(common::IndexMap::Mode::add);

    // Create RHS form
    L = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_forms_L, {V}, {{"g", g}, {"v_n", v_n}}, {{"c0", c0}},
                                 {{dolfinx::fem::IntegralType::exterior_facet, &(*Meshtags)}}));

    stiff = std::make_shared<StiffnessSpectral<double, P>>(V);
    b = std::make_shared<la::Vector<double>>(index_map, bs);
    _b = b->mutable_array();
  }

  // Set the initial values of u and v, i.e. u_0 and v_0
  void init() {
    u_n->x()->set(0.0);
    v_n->x()->set(0.0);
  }

  /// Evaluate du/dt = f0(t, u, v)
  /// @param[in] t Current time, i.e. tn
  /// @param[in] u Current u, i.e. un
  /// @param[in] v Current v, i.e. vn
  /// @param[out] result Result, i.e. dun/dtn
  void f0(double& t, std::shared_ptr<la::Vector<double>> u, std::shared_ptr<la::Vector<double>> v,
          std::shared_ptr<la::Vector<double>> result) {
    kernels::copy(*v, *result);
  }

  /// Evaluate dv/dt = f1(t, u, v)
  /// @param[in] t Current time, i.e. tn
  /// @param[in] u Current u, i.e. un
  /// @param[in] v Current v, i.e. vn
  /// @param[out] result Result, i.e. dvn/dtn
  void f1(double& t, std::shared_ptr<la::Vector<double>> u, std::shared_ptr<la::Vector<double>> v,
          std::shared_ptr<la::Vector<double>> result) {

    // Apply windowing
    if (t < T_ * alpha_) {
      window_ = 0.5 * (1.0 - cos(freq0_ * M_PI * t / alpha_));
    } else {
      window_ = 1.0;
    }

    // Update boundary condition
    std::fill(_g.begin(), _g.end(), window_ * p0_ * w0_ / c0_ * cos(w0_ * t));

    u->scatter_fwd();
    kernels::copy(*u, *u_n->x());

    v->scatter_fwd();
    kernels::copy(*v, *v_n->x());

    // Assemble RHS
    std::fill(_b.begin(), _b.end(), 0.0);
    stiff->operator()(*u_n->x(), *b);
    fem::assemble_vector(_b, *L);
    b->scatter_rev(common::IndexMap::Mode::add);

    // Solve
    // TODO: Divide is more expensive than multiply.
    // We should store the result of 1/m in a vector and apply and element wise vector
    // multiplication, since m doesn't change for linear wave propagation.
    {
      out = result->mutable_array();
      b_ = b->array();
      m_ = m->array();

      // Element wise division
      // out[i] = b[i]/m[i]
      std::transform(b_.begin(), b_.end(), m_.begin(), out.begin(),
                     [](const double& bi, const double& mi) { return bi / mi; });
    }
  }

  /// Runge-Kutta 4th order solver
  /// @param[in] startTime initial time of the solver
  /// @param[in] finalTime final time of the solver
  /// @param[in] timeStep  time step size of the solver
  void rk4(double& startTime, double& finalTime, double& timeStep) {
    
    // ------------------------------------------------------------------------
    // Computing function evaluation parameters

    std::string fname;

    // Grid parameters
    int Nx = 141;
    int Nz = 241;
    double tol = 1e-6;

    // Generate evaluate points
    xt::xarray<double> px = xt::linspace<double>(-0.035+tol, 0.035-tol, Nx);
    xt::xarray<double> pz = xt::linspace<double>(tol, 0.12-tol, Nz);
    auto [X, Z] = xt::meshgrid(px, pz);
    xt::xarray<double> x = xt::ravel(X);
    x.reshape({-1, 1});
    xt::xarray<double> y = xt::zeros<double>({Nx*Nz, 1});
    xt::xarray<double> z = xt::ravel(Z);
    z.reshape({-1, 1});
    auto pointsT = xt::hstack(xt::xtuple(x, y, z));

    // Compute evaluation parameters
    auto bb_tree = geometry::BoundingBoxTree(*mesh, mesh->topology().dim());
    auto cell_candidates = compute_collisions(bb_tree, pointsT);
    auto colliding_cells = geometry::compute_colliding_cells(
        *mesh, cell_candidates, pointsT);

    std::vector<int> cells;
    xt::xtensor<double, 2>::shape_type sh0 = {1, 3};
    auto points_on_proc = xt::empty<double>(sh0);
    for (int i = 0; i < Nx*Nz; i++) {
      auto link = colliding_cells.links(i);
      if (link.size() > 0) {
        auto p = xt::view(pointsT, i, xt::newaxis(), xt::all());
        points_on_proc = xt::vstack(xt::xtuple(points_on_proc, p));
        cells.push_back(link[0]);
      }
    }

    points_on_proc = xt::view(points_on_proc, xt::drop(0), xt::all());
    std::size_t lsize = points_on_proc.shape(0);
    xt::xtensor<double, 2> u_eval({lsize, 1});

    double* uval;
    double* pval = points_on_proc.data();

    // Time-stepping parameters
    double t = startTime;
    double tf = finalTime;
    double dt = timeStep;
    int step = 0;
    int nstep = (finalTime - startTime) / timeStep + 1;
    int numStepPerPeriod = T_ / dt + 1;
    int nstep_period = 0;

    // Time-stepping vectors
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

    while (t < tf) {
      dt = std::min(dt, tf - t);

      // Store solution at start of time step
      kernels::copy(*u_, *u0);
      kernels::copy(*v_, *v0);

      // Runge-Kutta step
      for (int i = 0; i < n_RK; i++) {
        kernels::copy(*u0, *un);
        kernels::copy(*v0, *vn);

        kernels::axpy(*un, dt * a_runge(i), *ku, *un);
        kernels::axpy(*vn, dt * a_runge(i), *kv, *vn);

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

      if (step % 200 == 0) {
        if (rank == 0) {
          std::cout << "t: " << t << ",\t Steps: " << step << "/" << nstep << std::endl;
        }
      }

      // Collect data for one period
      if (t > 0.12 / c0_ + 6.0 / freq0_ && nstep_period < numStepPerPeriod) {
        kernels::copy(*u_, *u_n->x());
        u_n->x()->scatter_fwd();

        // Function evaluation
        u_n->eval(points_on_proc, cells, u_eval);
        uval = u_eval.data();
        MPI_Barrier(MPI_COMM_WORLD);

        // Write evaluation from each process to a single text file
        for (int i = 0; i < size; i++) {
          if (rank == i) {
            fname = "/home/mabm4/rds/data/pressure_on_plane_" + 
                    std::to_string(nstep_period) + ".txt";
            std::ofstream txt_file(fname, std::ios_base::app);
            for (std::size_t i = 0; i < lsize; i++) {
              txt_file << *(pval + 3 * i) << ","
                       << *(pval + 3 * i + 2) << ","
                       << *(uval + i) << std::endl;
            }
            txt_file.close();
          }
          MPI_Barrier(MPI_COMM_WORLD);
        }
        nstep_period++;
      }
    }

    // Prepare solution at final time
    kernels::copy(*u_, *u_n->x());
    kernels::copy(*v_, *v_n->x());
    u_n->x()->scatter_fwd();
    v_n->x()->scatter_fwd();
  }

  std::size_t num_dofs() const { return V->dofmap()->index_map->size_global(); }

private:
  int rank, size; // MPI rank and size
  double c0_;     // speed of sound (m/s)
  double freq0_;  // source frequency (Hz)
  double p0_;     // pressure amplitude (Pa)
  double w0_;     // angular frequency (rad/s)
  double T_;      // period (s)
  double alpha_;
  double window_;

  std::shared_ptr<mesh::Mesh> mesh;
  std::shared_ptr<fem::FunctionSpace> V;
  std::shared_ptr<fem::Constant<double>> c0;
  std::shared_ptr<fem::Function<double>> u, v, g, u_n, v_n;
  std::shared_ptr<fem::Form<double>> a, L;
  std::shared_ptr<la::Vector<double>> m, b;

  xtl::span<double> _g, out;
  xtl::span<const double> m_, b_;
  xtl::span<double> _m, _b;

  std::shared_ptr<const common::IndexMap> index_map;
  int bs;

  std::shared_ptr<MassSpectral<double, P>> mass;
  std::shared_ptr<StiffnessSpectral<double, P>> stiff;
};