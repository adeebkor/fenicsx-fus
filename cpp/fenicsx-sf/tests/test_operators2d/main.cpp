// ==================
// Tests operators 2D
// ==================
// Copyright (C) Adeeb Arif Kor

#include "forms.h"
#include "spectral_op.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>

#define T_MPI MPI_DOUBLE

using namespace dolfinx;
using T = double;

int main(int argc, char* argv[]) 
{

  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Set polynomial degree
    const int P = 4;

    // Define mesh order
    const int G = 1;

    /*
    // Create mesh and function space
    const std::size_t N = 20;
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh = std::make_shared<mesh::Mesh>(
      mesh::create_rectangle(
        MPI_COMM_WORLD,
        {{{0.0, 0.0}, {1.0, 1.0}}},
        {N, N},
        mesh::CellType::quadrilateral,
        part));
    */

    // Read mesh and tags
    auto element = fem::CoordinateElement(mesh::CellType::quadrilateral, G);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "../mesh_1/mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(
      fmesh.read_mesh(element, mesh::GhostMode::none, "quad"));
    mesh->topology().create_connectivity(1, 2);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(mesh, "quad_cells"));

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_m1, "u", mesh));

    // Get index map and block size
    auto index_map = V->dofmap()->index_map;

    // Create functions
    auto u = std::make_shared<fem::Function<T>>(V);
    std::span<T> u_ = u->x()->mutable_array();
    std::fill(u_.begin(), u_.end(), 1.0);

    auto u_n = std::make_shared<fem::Function<T>>(V);
    std::span<T> un_ = u_n->x()->mutable_array();
    std::fill(un_.begin(), un_.end(), 2.0);

    auto v_n = std::make_shared<fem::Function<T>>(V);
    v_n->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> v_n(x.extent(1));
        
        for (std::size_t p = 0; p < x.extent(1); ++p)
          v_n[p] = std::cos(x(0, p)) * std::sin(std::numbers::pi * x(1, p));

        return {v_n, {v_n.size()}};
      });

      auto w_n = std::make_shared<fem::Function<T>>(V);
      std::transform(u_n->x()->array().begin(), u_n->x()->array().end(),
                     w_n->x()->mutable_array().begin(),
                     [&](const T& vx) { return vx * vx; }); 

    // Create DG functions
    auto V_DG = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_m1, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);
    auto delta0 = std::make_shared<fem::Function<T>>(V_DG);
    auto beta0 = std::make_shared<fem::Function<T>>(V_DG);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::span<T> delta0_ = delta0->x()->mutable_array();
    std::span<T> beta0_ = beta0->x()->mutable_array();
    
    std::fill(c0_.begin(), c0_.end(), 1.5);
    std::fill(rho0_.begin(), rho0_.end(), 1);
    std::fill(delta0_.begin(), delta0_.end(), 10);
    std::fill(beta0_.begin(), beta0_.end(), 10);

    // ------------------------------------------------------------------------
    // M1 coefficients
    std::vector<T> m1_coeffs(c0_.size());
    for (std::size_t i = 0; i < m1_coeffs.size(); ++i)
      m1_coeffs[i] = 1.0 / rho0_[i] / c0_[i] / c0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx m1 vector
    auto M1 = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_m1, {V}, 
                               {{"u", u}, {"c0", c0}, {"rho0", rho0}}, 
                               {}, {}));
    
    auto m1 = std::make_shared<fem::Function<T>>(V);
    fem::assemble_vector(m1->x()->mutable_array(), *M1);
    m1->x()->scatter_rev(std::plus<T>());

    auto m1_ = m1->x()->array();

    // ------------------------------------------------------------------------
    // Compute spectral mass 1 vector
    MassSpectral2D<T, P> mass_spectral_1(V);

    auto ms1 = std::make_shared<fem::Function<T>>(V);
    mass_spectral_1(*u->x(), m1_coeffs, *ms1->x());
    ms1->x()->scatter_rev(std::plus<T>());

    auto ms1_ = ms1->x()->array();

    // ------------------------------------------------------------------------
    // Print the first 10 values
    /*
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << m1_[i] << " " << ms1_[i] << "\n";
    */

    // ------------------------------------------------------------------------
    // Equality check (Mass 1)

    auto Em1 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_E, {}, {{"f0", m1}, {"f1", ms1}}, {}, {}, mesh));
    T error_m1 = fem::assemble_scalar(*Em1);
    T error_m1_sum;
    MPI_Reduce(&error_m1, &error_m1_sum, 1, T_MPI, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::cout << "Relative L2 error (mass 1): " << error_m1_sum << std::endl;
    }

    // ------------------------------------------------------------------------
    // M2 coefficients
    std::vector<T> m2_coeffs(c0_.size());
    for (std::size_t i = 0; i < m2_coeffs.size(); ++i)
      m2_coeffs[i] = - 2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] / c0_[i] / c0_[i] / c0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx m2 vector
    auto M2 = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_m2, {V}, 
                               {{"u", u}, {"u_n", u_n}, 
                                {"c0", c0}, {"rho0", rho0}, {"beta0", beta0}}, 
                               {}, {}));
    
    auto m2 = std::make_shared<fem::Function<T>>(V);
    fem::assemble_vector(m2->x()->mutable_array(), *M2);
    m2->x()->scatter_rev(std::plus<T>());

    auto m2_ = m2->x()->array();

    // ------------------------------------------------------------------------
    // Compute spectral mass 2 vector
    MassSpectral2D<T, P> mass_spectral_2(V);

    auto ms2 = std::make_shared<fem::Function<T>>(V);
    mass_spectral_2(*u_n->x(), m2_coeffs, *ms2->x());
    ms2->x()->scatter_rev(std::plus<T>());

    auto ms2_ = ms2->x()->array();

    // ------------------------------------------------------------------------
    // Print the first 10 values
    /*
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << m2_[i] << " " << ms2_[i] << "\n";
    */

    // ------------------------------------------------------------------------
    // Equality check (Mass 2)

    auto Em2 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_E, {}, {{"f0", m2}, {"f1", ms2}}, {}, {}, mesh));
    T error_m2 = fem::assemble_scalar(*Em2);
    T error_m2_sum;
    MPI_Reduce(&error_m2, &error_m2_sum, 1, T_MPI, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::cout << "Relative L2 error (mass 2): " << error_m2_sum << std::endl;
    }

    // ------------------------------------------------------------------------
    // M3 coefficients
    std::vector<T> m3_coeffs(c0_.size());
    for (std::size_t i = 0; i < m3_coeffs.size(); ++i)
      m3_coeffs[i] = 2.0 * beta0_[i] / rho0_[i] / rho0_[i] / c0_[i] / c0_[i] / c0_[i] / c0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx m3 vector
    auto M3 = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_m3, {V}, 
                               {{"u_n", u_n}, 
                                {"c0", c0}, {"rho0", rho0}, {"beta0", beta0}}, 
                               {}, {}));
    
    auto m3 = std::make_shared<fem::Function<T>>(V);
    fem::assemble_vector(m3->x()->mutable_array(), *M3);
    m3->x()->scatter_rev(std::plus<T>());

    auto m3_ = m3->x()->array();

    // ------------------------------------------------------------------------
    // Compute spectral mass 3 vector
    MassSpectral2D<T, P> mass_spectral_3(V);

    auto ms3 = std::make_shared<fem::Function<T>>(V);
    mass_spectral_3(*w_n->x(), m3_coeffs, *ms3->x());
    ms3->x()->scatter_rev(std::plus<T>());

    auto ms3_ = ms3->x()->array();

    // ------------------------------------------------------------------------
    // Print the first 10 values
    /*
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << m3_[i] << " " << ms3_[i] << "\n";
    */

    // ------------------------------------------------------------------------
    // Equality check (Mass 3)

    auto Em3 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_E, {}, {{"f0", m3}, {"f1", ms3}}, {}, {}, mesh));
    T error_m3 = fem::assemble_scalar(*Em3);
    T error_m3_sum;
    MPI_Reduce(&error_m3, &error_m3_sum, 1, T_MPI, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::cout << "Relative L2 error (mass 3): " << error_m3_sum << std::endl;
    }

    // ------------------------------------------------------------------------
    // S1 coefficients
    std::vector<T> s1_coeffs(c0_.size());
    for (std::size_t i = 0; i < s1_coeffs.size(); ++i)
      s1_coeffs[i] = - 1.0 / rho0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx s1 vector
    auto s1 = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_b1, {V},
                               {{"v_n", v_n}, {"rho0", rho0}},
                               {}, {}));

    auto b1 = std::make_shared<fem::Function<T>>(V);
    fem::assemble_vector(b1->x()->mutable_array(), *s1);
    b1->x()->scatter_rev(std::plus<T>());

    auto b1_ = b1->x()->array();

    // ------------------------------------------------------------------------
    // Compute spectral stiffness 1 vector
    StiffnessSpectral2D<T, P> stiffness_spectral_1(V);

    auto bs1 = std::make_shared<fem::Function<T>>(V);
    stiffness_spectral_1(*v_n->x(), s1_coeffs, *bs1->x());
    bs1->x()->scatter_rev(std::plus<T>());

    auto bs1_ = bs1->x()->array();

    // ------------------------------------------------------------------------
    // Print the first 10 values
    /*
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << b1_[i] << " " << bs1_[i] << "\n";
    */

    // ------------------------------------------------------------------------
    // Equality check (Stiffness 1)

    auto Es1 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_E, {}, {{"f0", b1}, {"f1", bs1}}, {}, {}, mesh));
    T error_s1 = fem::assemble_scalar(*Es1);
    T error_s1_sum;
    MPI_Reduce(&error_s1, &error_s1_sum, 1, T_MPI, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::cout << "Relative L2 error (stiffness 1): " << error_s1_sum << std::endl;
    }

    // ------------------------------------------------------------------------
    // S2 coefficients
    std::vector<T> s2_coeffs(c0_.size());
    for (std::size_t i = 0; i < s2_coeffs.size(); ++i)
      s2_coeffs[i] = - delta0_[i] / rho0_[i] / c0_[i] / c0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx s1 vector
    auto s2 = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_b2, {V},
                               {{"v_n", v_n}, 
                                {"c0", c0}, {"rho0", rho0}, {"delta0", delta0}},
                               {}, {}));

    auto b2 = std::make_shared<fem::Function<T>>(V);
    fem::assemble_vector(b2->x()->mutable_array(), *s2);
    b2->x()->scatter_rev(std::plus<T>());

    auto b2_ = b2->x()->array();

    // ------------------------------------------------------------------------
    // Compute spectral stiffness 2 vector
    StiffnessSpectral2D<T, P> stiffness_spectral_2(V);

    auto bs2 = std::make_shared<fem::Function<T>>(V);
    stiffness_spectral_2(*v_n->x(), s2_coeffs, *bs2->x());
    bs2->x()->scatter_rev(std::plus<T>());

    auto bs2_ = bs2->x()->array();

    // ------------------------------------------------------------------------
    // Print the first 10 values
    /*
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << b2_[i] << " " << bs2_[i] << "\n";
    */

    // ------------------------------------------------------------------------
    // Equality check (Stiffness 2)

    auto Es2 = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_E, {}, {{"f0", b2}, {"f1", bs2}}, {}, {}, mesh));
    T error_s2 = fem::assemble_scalar(*Es2);
    T error_s2_sum;
    MPI_Reduce(&error_s2, &error_s2_sum, 1, T_MPI, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::cout << "Relative L2 error (stiffness 2): " << error_s2_sum << std::endl;
    }
  }

  PetscFinalize();
}