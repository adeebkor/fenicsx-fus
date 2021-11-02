#include "forms.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <typeinfo>


void init(tcb::span<double> un, tcb::span<double> vn)
{
    std::fill(un.begin(), un.end(), 0.0);
    std::fill(vn.begin(), vn.end(), 0.0);
}

void update_source(float& t, float& p0, float& w0, float& c0, tcb::span<double> gn)
{
    std::fill(gn.begin(), gn.end(), p0*w0/c0*cos(w0*t));
}

void solve_ibvp()
{

}


int main(int argc, char* argv[])
{
    common::subsystem::init_logging(argc, argv);
    common::subsystem::init_petsc(argc, argv);

    // Material parameters
    float c0_ = 1.0;  // speed of sound (m/s)
    float rho0_ = 1.0;  // density of medium (kg/m^3)
    float beta_ = 0.01;  // coefficient of nonlinearity
    float delta_ = 0.001;  // diffusivity of sound

    // Source parameters
    float f0_ = 10.0;  // source frequency (Hz)
    float w0_ = 2.0 * M_PI * f0_;  // angular frequency (rad/s)
    float u0_ = 1.0;  // velocity amplitude (m/s)
    float p0_ = rho0_*c0_*u0_;  // pressure amplitude (Pa)

    // Domain parameters
    float xsh = rho0_*pow(c0_, 3)/beta_/p0_/w0_;  // shock formation distance (m)
    float length = 1.0;  // domain length (m)

    // Physical parameters
    float lambda = c0_/f0_;  // wavelength (m)
    float k = 2.0 * M_PI / lambda;  // wavenumber (m^-1)

    // FE parameters
    int degree = 4;  // degree of basis function

    // Mesh parameters
    int epw = 4;  // number of element per wavelength
    float nw = length / lambda;  // number of waves in the domain
    unsigned long nx = epw * nw + 1;  // total number of element in x-direction
    float h = sqrt(2 * pow(length / nx, 2));  // mesh size

    // Temporal parameters
    float CFL = 0.8;
    float dt = CFL * h / (c0_ * pow(degree, 2));

    std::cout << dt << std::endl;

    // Read mesh and mesh tags
    auto element = fem::CoordinateElement(mesh::CellType::quadrilateral, 1);
    auto xdmf = io::XDMFFile(MPI_COMM_WORLD, "../rectangle_dolfinx.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(xdmf.read_mesh(
        element, mesh::GhostMode::none, "rectangle"));
    mesh->topology().create_connectivity(1, 2);
    auto mt = std::make_shared<mesh::MeshTags<std::int32_t>>(
        xdmf.read_meshtags(mesh, "rectangle_edge"));

    // Define the function space
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_forms_a, "u", mesh));

    std::int64_t dof = V->dofmap()->index_map->size_global();
    std::cout << "Degrees of freedom: " << dof << std::endl;

    // Define functions
    auto c0 = std::make_shared<fem::Constant<PetscScalar>>(c0_);
    auto u = std::make_shared<fem::Function<PetscScalar>>(V);
    auto u_n = std::make_shared<fem::Function<PetscScalar>>(V);
    auto v_n = std::make_shared<fem::Function<PetscScalar>>(V);
    auto g = std::make_shared<fem::Function<PetscScalar>>(V);

    auto a = std::make_shared<fem::Form<PetscScalar>>(
        fem::create_form<PetscScalar>(*form_forms_a, {V},
                                      {{"u", u}}, {}, {}));
    auto L = std::make_shared<fem::Form<PetscScalar>>(
        fem::create_form<PetscScalar>(*form_forms_L, {V},
                                      {{"u_n", u_n}, {"g", g}, {"v_n", v_n}},
                                      {{"c0", c0}},
                                      {}));

    tcb::span<double> vec = u->x()->mutable_array();  // create a reference to RHS name vec1
    std::fill(vec.begin(), vec.end(), 1.0);  // fill the vector with ones

    init(u_n->x()->mutable_array(), v_n->x()->mutable_array());

    la::PETScVector m(*a->function_spaces()[0]->dofmap()->index_map,
                      a->function_spaces()[0]->dofmap()->index_map_bs());
    la::PETScVector b(*L->function_spaces()[0]->dofmap()->index_map,
                      L->function_spaces()[0]->dofmap()->index_map_bs());

    VecSet(m.vec(), 0.0);
    VecSet(b.vec(), 0.0);

    fem::assemble_vector_petsc(m.vec(), *a);
    fem::assemble_vector_petsc(b.vec(), *L);

    // update_source(t, p0_, w0_, c0_, g->x()->mutable_array());
    
    // std::cout << typeid(u_n->vector()).name() << std::endl;

}