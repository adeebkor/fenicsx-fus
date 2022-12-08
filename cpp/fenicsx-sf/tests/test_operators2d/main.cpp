// ==================
// Tests operators 2D
// ==================
// Copyright (C) Adeeb Arif Kor

#include "forms.h"
#include "spectral_op.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>

using namespace dolfinx;

int main(int argc, char* argv[]) 
{
  
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {

    // MPI

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Define polynomial degree
    const int P = 4;

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

    /*
    // Read mesh and tags
    auto element = fem::CoordinateElement(mesh::CellType::quadrilateral, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "../mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(
      fmesh.read_mesh(element, mesh::GhostMode::none, "planewave_2d_5"));
    mesh->topology().create_connectivity(1, 2);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(mesh, "planewave_2d_5_cells"));
    */

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_m, "u", mesh));

    // Get index map and block size
    auto index_map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    // Create input function
    auto u = std::make_shared<fem::Function<double>>(V);
    u->interpolate(
      [](auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
      {
        std::vector<double> u(x.extent(1));
        
        for (std::size_t p = 0; p < x.extent(1); ++p)
          u[p] = std::sin(x(0, p)) * std::cos(std::numbers::pi * x(1, p));

        return {u, {u.size()}};
      });

    // Create DG functions
    auto V_DG = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_m, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<double>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<double>>(V_DG);

    std::span<double> c0_ = c0->x()->mutable_array();
    std::span<double> rho0_ = rho0->x()->mutable_array();
    
    std::fill(c0_.begin(), c0_.end(), 1500);
    std::fill(rho0_.begin(), rho0_.end(), 1000);

    // ------------------------------------------------------------------------
    // Mass coefficients
    std::vector<double> m_coeffs(c0_.size());
    for (std::size_t i = 0; i < m_coeffs.size(); ++i)
      m_coeffs[i] = 1.0 / rho0_[i] / c0_[i] / c0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx mass vector
    auto m = std::make_shared<fem::Form<double>>(
      fem::create_form<double>(*form_forms_m, {V}, 
                               {{"u", u}, {"c0", c0}, {"rho0", rho0}}, 
                               {}, {}));
    
    la::Vector<double> m0(index_map, bs);
    fem::assemble_vector(m0.mutable_array(), *m);
    m0.scatter_rev(std::plus<double>());

    // ------------------------------------------------------------------------
    // Compute spectral mass vector
    MassSpectral2D<double, P> mass_spectral(V);

    la::Vector<double> m1(index_map, bs);
    mass_spectral(*u->x(), m_coeffs, m1);
    m1.scatter_rev(std::plus<double>());

    // ------------------------------------------------------------------------
    // Print the first 10 values
    /*
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << m0.array()[i] << " " << m1.array()[i] << "\n";
    */

    // ------------------------------------------------------------------------
    // Equality check (Mass)

    float rel_err1 = 0;

    for (std::size_t i = 0; i < m0.array().size(); ++i)
    {
      rel_err1 += (m1.array()[i] - m0.array()[i]) 
        * (m1.array()[i] - m0.array()[i])
        / (m0.array()[i] * m0.array()[i] + 1e-10);
    }

    std::cout << "Relative L2 error (mass), " 
              << "PROC" << mpi_rank << " : " << rel_err1 << std::endl;

    // ------------------------------------------------------------------------
    // Stiffness coefficients
    std::vector<double> s_coeffs(c0_.size());
    for (std::size_t i = 0; i < s_coeffs.size(); ++i)
      s_coeffs[i] = - 1.0 / rho0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx stiffness vector
    auto s = std::make_shared<fem::Form<double>>(
      fem::create_form<double>(*form_forms_s, {V},
                               {{"u", u}, {"rho0", rho0}},
                               {}, {}));

    la::Vector<double> s0(index_map, bs);
    fem::assemble_vector(s0.mutable_array(), *s);
    s0.scatter_rev(std::plus<double>());

    // ------------------------------------------------------------------------
    // Compute spectral stiffness vector
    StiffnessSpectral2D<double, P> stiffness_spectral(V);

    la::Vector<double> s1(index_map, bs);
    stiffness_spectral(*u->x(), s_coeffs, s1);
    s1.scatter_rev(std::plus<double>());

    // ------------------------------------------------------------------------
    // Print the first 10 values
    /*
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << s0.array()[i] << " " << s1.array()[i] << "\n";
    */

    // ------------------------------------------------------------------------
    // Equality check (Stiffness)

    float rel_err2 = 0;

    for (std::size_t i = 0; i < s0.array().size(); ++i)
    {
      rel_err2 += (s1.array()[i] - s0.array()[i]) 
        * (s1.array()[i] - s0.array()[i])
        / (s0.array()[i] * s0.array()[i] + 1e-10);
    }

    std::cout << "Relative L2 error (stiffness), " 
              << "PROC" << mpi_rank << " : " << rel_err2 << std::endl;
  }

  PetscFinalize();
}