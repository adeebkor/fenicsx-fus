// ==================
// Tests operators 3D
// ==================
// Copyright (C) Adeeb Arif Kor

#include "forms.h"
#include "spectral_op.hpp"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>

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

    /*
    // Create mesh and function space
    const std::size_t N = 20;
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
    auto mesh = std::make_shared<mesh::Mesh>(
      mesh::create_box(
        MPI_COMM_WORLD,
        {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
        {N, N, N},
        mesh::CellType::hexahedron,
        part));
    */

    // Read mesh and tags
    auto element = fem::CoordinateElement(mesh::CellType::hexahedron, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, "../mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh>(
      fmesh.read_mesh(element, mesh::GhostMode::none, "hex"));
    mesh->topology().create_connectivity(2, 3);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
      fmesh.read_meshtags(mesh, "hex_cells"));

    // Create function space
    auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_m, "u", mesh));

    // Get index map and block size
    auto index_map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    // Create input function
    auto u = std::make_shared<fem::Function<T>>(V);
    u->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> u(x.extent(1));
        
        for (std::size_t p = 0; p < x.extent(1); ++p)
          u[p] = std::sin(x(0, p)) * std::cos(std::numbers::pi * x(1, p));

        return {u, {u.size()}};
      });

    // Create DG functions
    auto V_DG = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(functionspace_form_forms_m, "c0", mesh));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::span<T> rho0_ = rho0->x()->mutable_array();
    
    std::fill(c0_.begin(), c0_.end(), 1.5e-3);
    std::fill(rho0_.begin(), rho0_.end(), 1e-3);

    // ------------------------------------------------------------------------
    // Stiffness coefficients
    std::vector<T> s_coeffs(c0_.size());
    for (std::size_t i = 0; i < s_coeffs.size(); ++i)
      s_coeffs[i] = - 1.0 / rho0_[i];

    // ------------------------------------------------------------------------
    // Compute dolfinx stiffness vector
    auto s = std::make_shared<fem::Form<T>>(
      fem::create_form<T>(*form_forms_s, {V},
                               {{"u", u}, {"rho0", rho0}},
                               {}, {}));

    // la::Vector<T> s0(index_map, bs);
    // fem::assemble_vector(s0.mutable_array(), *s);
    // s0.scatter_rev(std::plus<T>());

    auto s0 = std::make_shared<fem::Function<T>>(V);
    fem::assemble_vector(s0->x()->mutable_array(), *s);
    s0->x()->scatter_rev(std::plus<T>());

    auto s0_ = s0->x()->array();

    // ------------------------------------------------------------------------
    // Compute spectral stiffness vector
    StiffnessSpectral3D<T, P> stiffness_spectral(V);

    // la::Vector<T> s1(index_map, bs);
    // stiffness_spectral(*u->x(), s_coeffs, s1);
    // s1.scatter_rev(std::plus<T>());

    auto s1 = std::make_shared<fem::Function<T>>(V);
    stiffness_spectral(*u->x(), s_coeffs, *s1->x());
    s1->x()->scatter_rev(std::plus<T>());

    auto s1_ = s1->x()->array();

    // ------------------------------------------------------------------------
    // Print the first 10 values
    
    for (std::size_t i = 0; i < 10; ++i)
      std::cout << s0_[i] << " " << s1_[i] << "\n";
    

    // ------------------------------------------------------------------------
    // Equality check (Stiffness)

    // float rel_err2 = 0;

    // for (std::size_t i = 0; i < s0.array().size(); ++i)
    // {
    //   rel_err2 += (s1.array()[i] - s0.array()[i]) 
    //     * (s1.array()[i] - s0.array()[i])
    //     / (s0.array()[i] * s0.array()[i] + 1e-10);
    // }

    // std::cout << "Relative L2 error (stiffness), " 
    //           << "PROC" << mpi_rank << " : " << rel_err2 << std::endl;

    auto Es = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_forms_E, {}, {{"f0", s0}, {"f1", s1}}, {}, {}, mesh));
    T error_s = fem::assemble_scalar(*Es);

    std::cout << "Relative L2 error (stiffness), " 
              << "PROC" << mpi_rank << " : " << error_s << std::endl;
  }

  PetscFinalize();
}