#include "forms.h"

#include <dolfinx.h>
// #include <dolfinx/fem/petsc.h>
#include <dolfinx/la/Vector.h>
#include <memory>

using namespace dolfinx;

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

    // TODO: Add comments about this operation. Is this the Mass matrix operator?
    m = std::make_shared<la::Vector<double>>(*index_map, bs);
    xtl::span<double> m_data = m->mutable_array();
    std::fill(m_data.begin(), m_data.end(), 0);
    fem::assemble_vector(m_data, *a);

    // Create RHS form
    L = std::make_shared<fem::Form<double>>(fem::create_form<double>(
        *form_forms_L, {V}, {{"u_n", u_n}, {"g", g}, {"v_n", v_n}}, {{"c0", c0}},
        {{dolfinx::fem::IntegralType::exterior_facet, &(*Meshtags)}}));

    // Allocate memory for the RHS
    b = std::make_shared<la::Vector<double>>(*index_map, bs);
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

    xtl::span<const double> in = v_f0->array();
    xtl::span<double> out = result_f0->mutable_array();
    std::copy(in.cbegin(), in.cend(), out.begin());
  }

  /// TODO: ADD documentation
  void f1(double& t, std::shared_ptr<la::PETScVector> u_f1, std::shared_ptr<la::PETScVector> v_f1,
          std::shared_ptr<la::PETScVector> result_f1) {

    // Apply windowing
    if (t < T_ * alpha_) {
      window_ = 0.5 * (1.0 - cos(freq0_ * M_PI * t / alpha_));
    } else {
      window_ = 1.0;
    }

    // Update boundary condition
    g_vec = g->x()->mutable_array();
    std::fill(g_vec.begin(), g_vec.end(), window_ * p0_ * w0_ / c0_ * cos(w0_ * t));

    // Update fields
    VecCopy(u_f1->vec(), u_n->vector());
    // u_n->x()->scatter_fwd();
    VecGhostUpdateBegin(u_n->vector(), INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(u_n->vector(), INSERT_VALUES, SCATTER_FORWARD);

    VecCopy(v_f1->vec(), v_n->vector());
    // v_n->x()->scatter_fwd();
    VecGhostUpdateBegin(v_n->vector(), INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(v_n->vector(), INSERT_VALUES, SCATTER_FORWARD);

    // const auto L_coeffs = pack_coefficients(*L);
    // const auto L_consts = pack_constants(*L);

    // Assemble RHS
    VecSet(b->vec(), 0.0);
    fem::assemble_vector_petsc(b->vec(), *L); //, L_consts, L_coeffs);
    // VecGhostUpdateBegin(b->vec(), ADD_VALUES, SCATTER_REVERSE);
    // VecGhostUpdateEnd(b->vec(), ADD_VALUES, SCATTER_REVERSE);

    PetscInt n = 1;
    const PetscInt ix[1] = {20};
    PetscScalar y[1], z[1], p[1], q[1];
    VecGetValues(b->vec(), n, ix, y);
    VecGetValues(m->vec(), n, ix, z);
    VecGetValues(u_f1->vec(), n, ix, p);
    VecGetValues(v_f1->vec(), n, ix, q);
    std::cout << "g[10] = " << g->x()->mutable_array()[20] << std::endl;
    std::cout << "u[10] = " << p[0] << std::endl;
    std::cout << "u_n[10] = " << u_n->x()->mutable_array()[20] << std::endl;
    std::cout << "v[10] = " << q[0] << std::endl;
    std::cout << "v_n[10] = " << v_n->x()->mutable_array()[20] << std::endl;
    std::cout << "m[10] = " << z[0] << std::endl;
    std::cout << "b[10] = " << y[0] << std::endl;
    std::getchar();

    // Solve
    VecPointwiseDivide(result_f1->vec(), b->vec(), m->vec());
  }

  void solve_ibvp(double& startTime, double& finalTime, double& timeStep) {
    double t = startTime;
    double tf = finalTime;
    double dt = timeStep;
    int step = 0;
    int nstep = (finalTime - startTime) / timeStep + 1;

    // Placeholder vectors at time step n
    un = std::make_shared<la::PETScVector>(*L->function_spaces()[0]->dofmap()->index_map,
                                           L->function_spaces()[0]->dofmap()->index_map_bs());
    vn = std::make_shared<la::PETScVector>(*L->function_spaces()[0]->dofmap()->index_map,
                                           L->function_spaces()[0]->dofmap()->index_map_bs());

    // Placeholder vectors at start of time step
    u0 = std::make_shared<la::PETScVector>(*L->function_spaces()[0]->dofmap()->index_map,
                                           L->function_spaces()[0]->dofmap()->index_map_bs());
    v0 = std::make_shared<la::PETScVector>(*L->function_spaces()[0]->dofmap()->index_map,
                                           L->function_spaces()[0]->dofmap()->index_map_bs());

    // Placeholder at k intermediate time step
    ku = std::make_shared<la::PETScVector>(*L->function_spaces()[0]->dofmap()->index_map,
                                           L->function_spaces()[0]->dofmap()->index_map_bs());
    kv = std::make_shared<la::PETScVector>(*L->function_spaces()[0]->dofmap()->index_map,
                                           L->function_spaces()[0]->dofmap()->index_map_bs());

    VecCopy(u_n->vector(), ku->vec());
    VecCopy(v_n->vector(), kv->vec());

    // Runge-Kutta timestepping data
    int n_RK = 4;
    xt::xarray<double> a_runge{0.0, 0.5, 0.5, 1.0};
    xt::xarray<double> b_runge{1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
    xt::xarray<double> c_runge{0.0, 0.5, 0.5, 1.0};

    // RK variables
    double alp;
    double tn;

    PetscInt n = 1;
    const PetscInt ix[1] = {10};
    PetscScalar y[1];
    PetscScalar z[1];
    PetscScalar x[1];

    while (t < tf) {
      dt = std::min(dt, tf - t);

      // Store solution at start of time step
      VecCopy(u_n->vector(), u0->vec());
      VecCopy(v_n->vector(), v0->vec());

      // Runge-Kutta step
      for (int i = 0; i < n_RK; i++) {

        VecCopy(u0->vec(), un->vec());
        VecCopy(v0->vec(), vn->vec());

        alp = dt * a_runge(i);
        VecAXPY(un->vec(), alp, ku->vec());
        VecAXPY(vn->vec(), alp, kv->vec());

        // RK time evaluation
        tn = t + c_runge(i) * dt;

        // Compute RHS vector
        kv->a f0(tn, un, vn, ku); // ku = vn; move pointer?
        f1(tn, un, vn, kv);
        VecGetValues(vn->vec(), n, ix, x);
        VecGetValues(ku->vec(), n, ix, y);
        VecGetValues(kv->vec(), n, ix, z);
        std::cout << "RK step: " << i << std::endl;

        // Update solution
        VecAXPY(u_n->vector(), dt * b_runge(i), ku->vec());
        VecAXPY(v_n->vector(), dt * b_runge(i), kv->vec());
      }

      // Update time
      t += dt;
      step += 1;

      if (step % 100 == 0) {
        PetscSynchronizedPrintf(MPI_COMM_WORLD, "%f %D/%D \n", t, step, nstep);
      }
    }
    std::cout << t << std::endl;
    std::cout.precision(15); // Set print precision
    dolfinx::io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write({*u_n}, 0.0);
  }
};