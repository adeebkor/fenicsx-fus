#include "forms.h"
#include <iostream>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/VTKFile.h>


class LinearGLL
{
    protected:
        int k_; // degree of basis function
        double c0_; // speed of sound (m/s)
        double freq0_; // source frequency (Hz)
        double p0_; // pressure amplitude (Pa)
        double w0_; // angular frequency (rad/s)
        double T_; // period (s)
        double alpha_;
        double window_;

        tcb::span<double> g_vec;

        std::shared_ptr<fem::Constant<double>> c0;
        std::shared_ptr<fem::Function<double>> u, v, g, u_n, v_n;
        std::shared_ptr<fem::Form<double>> a, L;
        std::shared_ptr<la::PETScVector> m, b;
        std::shared_ptr<la::PETScVector> un, vn, u0, v0, ku, kv;

    public:
        std::shared_ptr<fem::FunctionSpace> V;

        LinearGLL(std::shared_ptr<mesh::Mesh> Mesh,
                  std::shared_ptr<mesh::MeshTags<std::int32_t>> Meshtags,
                  int& degreeOfBasis, double& speedOfSound,
                  double& sourceFrequency, double& pressureAmplitude)
        {
            V = std::make_shared<fem::FunctionSpace>(
                fem::create_functionspace(
                    functionspace_form_forms_a, "u", Mesh));
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
            tcb::span<double> vec = u->x()->mutable_array();
            std::fill(vec.begin(), vec.end(), 1.0);

            a = std::make_shared<fem::Form<double>>(
                    fem::create_form<double>(
                        *form_forms_a, {V},
                        {{"u", u}}, {}, {}));

            m = std::make_shared<la::PETScVector>(
                    *a->function_spaces()[0]->dofmap()->index_map,
                    a->function_spaces()[0]->dofmap()->index_map_bs());

            VecSet(m->vec(), 0.0);
            fem::assemble_vector_petsc(m->vec(), *a);

            L = std::make_shared<fem::Form<double>>(
                    fem::create_form<double>(
                        *form_forms_L, {V},
                        {{"u_n", u_n}, {"g", g}, {"v_n", v_n}},
                        {{"c0", c0}},
                        {{dolfinx::fem::IntegralType::exterior_facet, &(*Meshtags)}}));
            
            b = std::make_shared<la::PETScVector>(
                *L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());

        }

        void init()
        {
            tcb::span<double> u_vec = u_n->x()->mutable_array();
            tcb::span<double> v_vec = v_n->x()->mutable_array();

            std::fill(u_vec.begin(), u_vec.end(), 0.0);
            std::fill(v_vec.begin(), v_vec.end(), 0.0); 
        }

        void f0(double& t, std::shared_ptr<la::PETScVector> u_f0,
                          std::shared_ptr<la::PETScVector> v_f0,
                          std::shared_ptr<la::PETScVector> result_f0)
        {
            VecCopy(v_f0->vec(), result_f0->vec());
        }

        void f1(double& t, std::shared_ptr<la::PETScVector> u_f1,
                          std::shared_ptr<la::PETScVector> v_f1,
                          std::shared_ptr<la::PETScVector> result_f1)
        {

            // Apply windowing
            if (t < T_ * alpha_)
            {
                window_ = 0.5 * (1.0 - cos(freq0_ * M_PI * t / alpha_));
            }
            else
            {
                window_ = 1.0;
            }

            // Update boundary condition
            g_vec = g->x()->mutable_array();
            std::fill(g_vec.begin(), g_vec.end(),
                      window_*p0_*w0_/c0_*cos(w0_*t));

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
            fem::assemble_vector_petsc(b->vec(), *L);//, L_consts, L_coeffs);
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

        void solve_ibvp(double& startTime, double& finalTime, double& timeStep)
        {
            double t = startTime;
            double tf = finalTime;
            double dt = timeStep;
            int step = 0;
            int nstep = (finalTime-startTime)/timeStep + 1;

            // Placeholder vectors at time step n
            un = std::make_shared<la::PETScVector>(
                *L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());
            vn = std::make_shared<la::PETScVector>(
                *L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());

            // Placeholder vectors at start of time step
            u0 = std::make_shared<la::PETScVector>(
                *L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());
            v0 = std::make_shared<la::PETScVector>(
                *L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());

            // Placeholder at k intermediate time step
            ku = std::make_shared<la::PETScVector>(
                *L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());
            kv = std::make_shared<la::PETScVector>(
                *L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());

            VecCopy(u_n->vector(), ku->vec());
            VecCopy(v_n->vector(), kv->vec());

            // Runge-Kutta timestepping data
            int n_RK = 4;
            xt::xarray<double> a_runge {0.0, 0.5, 0.5, 1.0};
            xt::xarray<double> b_runge {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
            xt::xarray<double> c_runge {0.0, 0.5, 0.5, 1.0};

            // RK variables
            double alp;
            double tn;

            PetscInt n = 1;
            const PetscInt ix[1] = {10};
            PetscScalar y[1];
            PetscScalar z[1]; 
            PetscScalar x[1]; 

            while (t < tf)
            {
                dt = std::min(dt, tf-t);
                
                // Store solution at start of time step
                VecCopy(u_n->vector(), u0->vec());
                VecCopy(v_n->vector(), v0->vec());

                // Runge-Kutta step
                for (int i=0; i<n_RK; i++)
                {
                    
                    VecCopy(u0->vec(), un->vec());
                    VecCopy(v0->vec(), vn->vec());

                    alp = dt * a_runge(i);
                    VecAXPY(un->vec(), alp, ku->vec());
                    VecAXPY(vn->vec(), alp, kv->vec());

                    // RK time evaluation
                    tn = t + c_runge(i)*dt;

                    // Compute RHS vector
                    std::cout << "RK step: " << i << std::endl;
                    f0(tn, un, vn, ku); // ku = vn; move pointer?
                    f1(tn, un, vn, kv);
                    VecGetValues(vn->vec(), n, ix, x);
                    VecGetValues(ku->vec(), n, ix, y);
                    VecGetValues(kv->vec(), n, ix, z);
                    // std::cout << "ku[10] = " << y[0] << std::endl;
                    // std::cout << "kv[10] = " << z[0] << std::endl;

                    // Update solution
                    VecAXPY(u_n->vector(), dt*b_runge(i), ku->vec());
                    VecAXPY(v_n->vector(), dt*b_runge(i), kv->vec());
                }

                // Update time
                t += dt;
                step += 1;

                if (step % 100 == 0)
                {
                    PetscSynchronizedPrintf(MPI_COMM_WORLD, "%f %D/%D \n", t, step, nstep);
                }

            }
            std::cout << t << std::endl;
            std::cout.precision(15); // Set print precision
            dolfinx::io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
            file.write({*u_n}, 0.0);
        }
};

int main(int argc, char* argv[])
{
    common::subsystem::init_logging(argc, argv);
    common::subsystem::init_petsc(argc, argv);

    std::cout.precision(15); // Set print precision

    // Material parameters
    double speedOfSound = 1.0;  // (m/s)
    double densityOfMedium = 1.0;  // (kg/m^3)
    double coeffOfNonlinearity = 0.01;
    double diffusivityOfSound = 0.001;

    // Source parameters
    double sourceFrequency = 10.0; // (Hz)
    double angularFrequency = 2.0 * M_PI * sourceFrequency;  // (rad/s)
    double velocityAmplitude = 1.0;  // (m/s)
    double pressureAmplitude = densityOfMedium*speedOfSound*
                              velocityAmplitude; // (Pa)

    // Domain parameters
    double shockFormationDistance = densityOfMedium*pow(speedOfSound, 3)/
                                   coeffOfNonlinearity/pressureAmplitude/
                                   angularFrequency;  // (m)
    double domainLength = 1.0;  // (m)

    // Physical parameters
    double wavelength = speedOfSound/sourceFrequency;  // (m)
    double wavenumber = 2.0 * M_PI / wavelength;  // (m^-1)

    // FE parameters
    int degreeOfBasis = 4;

    // Mesh parameters
    int elementPerWavelength = 4;
    double numberOfWaves = domainLength / wavelength;
    int numberOfElement = elementPerWavelength * numberOfWaves + 1;
    double meshSize = sqrt(2 * pow(domainLength / numberOfElement, 2));

    // Temporal parameters
    double CFL = 0.8;
    double timeStepSize = CFL * meshSize / 
                         (speedOfSound * pow(degreeOfBasis, 2));
    double startTime = 0.0;
    double finalTime = timeStepSize + domainLength / speedOfSound + 10.0 / sourceFrequency;
    
    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::quadrilateral, 1);
    auto xdmf = io::XDMFFile(MPI_COMM_WORLD, "../rectangle_dolfinx.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(xdmf.read_mesh(
        element, mesh::GhostMode::none, "rectangle"));
    mesh->topology().create_connectivity(1, 2);
    auto mt = std::make_shared<mesh::MeshTags<std::int32_t>>(
        xdmf.read_meshtags(mesh, "rectangle_edge"));
    
    // Model
    LinearGLL eqn(mesh, mt, degreeOfBasis, speedOfSound, sourceFrequency,
                  pressureAmplitude);

    std::cout << "Degrees of freedom: " << 
                 eqn.V->dofmap()->index_map->size_global() << std::endl;

    // RK solve
    eqn.init();
    eqn.solve_ibvp(startTime, finalTime, timeStepSize);

    common::subsystem::finalize_petsc();
    return 0;
}