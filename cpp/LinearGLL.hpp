#include "forms.h"

#include <dolfinx.h>
#include <dolfinx/la/Vector.h>
#include <memory>

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

class LinearGLL {
protected:
  int k_;        // degree of basis function
  double c0_;    // speed of sound (m/s)
  double freq0_; // source frequency (Hz)
  double p0_;    // pressure amplitude (Pa)
  double w0_;    // angular frequency (rad/s)
  double T_;     // period (s)
  double alpha_;
  double window_;

  std::shared_ptr<fem::Constant<double>> c0;
  std::shared_ptr<fem::Form<double>> a, L;
  std::shared_ptr<fem::Function<double>> u, v, g, u_n, v_n;
  std::shared_ptr<la::Vector<double>> m, b, un, vn, u0, v0, ku, kv;

public:
  std::shared_ptr<fem::FunctionSpace> V;

  LinearGLL(std::shared_ptr<mesh::Mesh> Mesh,
            std::shared_ptr<mesh::MeshTags<std::int32_t>> Meshtags, int& degreeOfBasis,
            double& speedOfSound, double& sourceFrequency, double& pressureAmplitude) {
    V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_forms_a, "u", Mesh));

    std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    c0 = std::make_shared<fem::Constant<double>>(speedOfSound);
    u = std::make_shared<fem::Function<double>>(V);
    v = std::make_shared<fem::Function<double>>(V);
    g = std::make_shared<fem::Function<double>>(V);
    u_n = std::make_shared<fem::Function<double>>(V);
    v_n = std::make_shared<fem::Function<double>>(V);

    // Physical parameters
    k_ = degreeOfBasis;
    c0_ = speedOfSound;
    freq0_ = sourceFrequency;
    p0_ = pressureAmplitude;
    w0_ = 2.0 * M_PI * freq0_;
    T_ = 1 / freq0_;
    alpha_ = 4;

    // Define variational formulation
    xtl::span<double> vec = u->x()->mutable_array();
    std::fill(vec.begin(), vec.end(), 1.0);

    a = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_forms_a, {V}, {{"u", u}}, {}, {}));

    // // TODO: Add comments about this operation. Is this the Mass matrix diagonal?
    m = std::make_shared<la::Vector<double>>(index_map, bs);
    xtl::span<double> m_data = m->mutable_array();
    std::fill(m_data.begin(), m_data.end(), 0);
    fem::assemble_vector(m_data, *a);

    // Create RHS form
    L = std::make_shared<fem::Form<double>>(fem::create_form<double>(
        *form_forms_L, {V}, {{"u_n", u_n}, {"g", g}, {"v_n", v_n}}, {{"c0", c0}},
        {{dolfinx::fem::IntegralType::exterior_facet, &(*Meshtags)}}));

    // Allocate memory for the RHS
    b = std::make_shared<la::Vector<double>>(index_map, bs);
  }

  /// TODO: Add documentation
  void init() {
    tcb::span<double> u_vec = u_n->x()->mutable_array();
    tcb::span<double> v_vec = v_n->x()->mutable_array();

    std::fill(u_vec.begin(), u_vec.end(), 0.0);
    std::fill(v_vec.begin(), v_vec.end(), 0.0);
  }

  /// TODO: ADD documentation
  void f0(double& t, std::shared_ptr<la::Vector<double>> u_f0,
          std::shared_ptr<la::Vector<double>> v_f0, std::shared_ptr<la::Vector<double>> result_f0) {
    kernels::copy(*v_f0, *result_f0);
  }

  /// TODO: ADD documentation
  void f1(double& t, std::shared_ptr<la::Vector<double>> u_f1,
          std::shared_ptr<la::Vector<double>> v_f1, std::shared_ptr<la::Vector<double>> result_f1) {

    // Apply windowing
    if (t < T_ * alpha_) {
      window_ = 0.5 * (1.0 - cos(freq0_ * M_PI * t / alpha_));
    } else {
      window_ = 1.0;
    }

    // Update boundary condition
    xtl::span<double> g_vec = g->x()->mutable_array();
    std::fill(g_vec.begin(), g_vec.end(), window_ * p0_ * w0_ / c0_ * cos(w0_ * t));

    u_f1->scatter_fwd();
    kernels::copy(*u_f1, *u_n->x());

    v_f1->scatter_fwd();
    kernels::copy(*v_f1, *v_n->x());

    // TODO: Compute coefficients

    // Assemble RHS
    tcb::span<double> b_ = b->mutable_array(); // Get underlying data
    std::fill(b_.begin(), b_.end(), 0);
    fem::assemble_vector(b_, *L);
    b->scatter_fwd();

    std::cout << "g[10] = " << g->x()->mutable_array()[20] << std::endl;
    std::cout << "u[10] = " << u_f1->array()[20] << std::endl;
    std::cout << "u_n[10] = " << u_n->x()->mutable_array()[20] << std::endl;
    std::cout << "v[10] = " << v_f1->array()[20] << std::endl;
    std::cout << "v_n[10] = " << v_n->x()->mutable_array()[20] << std::endl;
    std::cout << "m[10] = " << m->array()[20] << std::endl;
    std::cout << "b[10] = " << b->array()[20] << std::endl;
    std::getchar();

    // Solve
    // TODO: Divide is more expensive than multiply.
    // We should store the result of 1/m in a vector and apply and element wise vector
    // multiplication, since m doesn't change for linear wave propagation.
    {
      xtl::span<double> out = result_f1->mutable_array();
      xtl::span<const double> b_ = b->array();
      xtl::span<const double> m_ = m->array();

      // Element wise division
      // out[i] = b[i]/m[i]
      std::transform(b_.begin(), b_.end(), m_.begin(), out.begin(),
                     [](const double& bi, const double& mi) { return bi / mi; });
    }
  }

  void solve_ibvp(double& startTime, double& finalTime, double& timeStep) {
    double t = startTime;
    double tf = finalTime;
    double dt = timeStep;
    int step = 0;
    int nstep = (finalTime - startTime) / timeStep + 1;

    std::shared_ptr<const common::IndexMap> index_map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    // Placeholder vectors at time step n
    un = std::make_shared<la::Vector<double>>(index_map, bs);
    vn = std::make_shared<la::Vector<double>>(index_map, bs);

    // Placeholder vectors at start of time step
    u0 = std::make_shared<la::Vector<double>>(index_map, bs);
    v0 = std::make_shared<la::Vector<double>>(index_map, bs);

    // Placeholder at k intermediate time step
    ku = std::make_shared<la::Vector<double>>(index_map, bs);
    kv = std::make_shared<la::Vector<double>>(index_map, bs);

    kernels::copy(*u_n->x(), *ku);
    kernels::copy(*v_n->x(), *kv);

    // Runge-Kutta timestepping data
    int n_RK = 4;
    xt::xarray<double> a_runge{0.0, 0.5, 0.5, 1.0};
    xt::xarray<double> b_runge{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    xt::xarray<double> c_runge{0.0, 0.5, 0.5, 1.0};

    // RK variables
    double alp;
    double tn;

    int n = 1;
    while (t < tf) {
      dt = std::min(dt, tf - t);

      // Store solution at start of time step
      kernels::copy(*u_n->x(), *u0);
      kernels::copy(*v_n->x(), *v0);

      // Runge-Kutta step
      for (int i = 0; i < n_RK; i++) {
        kernels::copy(*u0, *un);
        kernels::copy(*v0, *vn);

        alp = dt * a_runge(i);
        kernels::axpy(*un, alp, *ku, *un);
        kernels::axpy(*vn, alp, *kv, *vn);

        // RK time evaluation
        tn = t + c_runge(i) * dt;

        // Compute RHS vector
        f0(tn, un, vn, ku);
        f1(tn, un, vn, kv);

        std::cout << "RK step: " << i << std::endl;

        // // Update solution
        kernels::axpy(*u_n->x(), dt * b_runge(i), *ku, *u_n->x());
        kernels::axpy(*v_n->x(), dt * b_runge(i), *kv, *v_n->x());
      }

      // Update time
      t += dt;
      step += 1;
    }
    std::cout << t << std::endl;
    std::cout.precision(15); // Set print precision
    dolfinx::io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write({*u_n}, 0.0);
  }
};