#pragma once

#include "forms.h"

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
class LinearGLLPenetrable {
public:
  LinearGLLPenetrable(std::shared_ptr<mesh::Mesh> Mesh,
                      std::shared_ptr<mesh::MeshTags<std::int32_t>> Meshtags,
                      double& speedOfSound_0, double& speedOfSound_1, 
                      double& sourceFrequency, double& pressureAmplitude) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mesh = Mesh;
    V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_forms_a, "u", Mesh));
    V_DG = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_forms_L, "c0", Mesh));

    index_map = V->dofmap()->index_map;
    bs = V->dofmap()->index_map_bs();

    c0 = std::make_shared<fem::Function<double>>(V_DG);
    u = std::make_shared<fem::Function<double>>(V);
    v = std::make_shared<fem::Function<double>>(V);
    g = std::make_shared<fem::Function<double>>(V);
    u_n = std::make_shared<fem::Function<double>>(V);
    v_n = std::make_shared<fem::Function<double>>(V);

    _g = g->x()->mutable_array();

    // Physical parameters
    c0_ = speedOfSound_0;
    c1_ = speedOfSound_1;
    freq0_ = sourceFrequency;
    p0_ = pressureAmplitude;
    w0_ = 2.0 * M_PI * freq0_;
    T_ = 1.0 / freq0_;
    alpha_ = 4.0;

  }

private:
  int rank, size;  // MPI rank and size
  double c0_, c1_; // speed of sound (m/s)
  double freq0_;   // source frequency (Hz)
  double p0_;      // pressure amplitude (Pa)
  double w0_;      // angular frequency (rad/s)
  double T_;       // period (s)
  double alpha_;
  double window_;

  std::shared_ptr<mesh::Mesh> mesh;
  std::shared_ptr<fem::FunctionSpace> V, V_DG;
  std::shared_ptr<fem::Function<double>> c0;
  std::shared_ptr<fem::Function<double>> u, v, g, u_n, v_n;
  std::shared_ptr<fem::Form<double>> a, L;
  std::shared_ptr<la::Vector<double>> m, b;

  xtl::span<double> _g, out;
  xtl::span<const double> m_, b_;
  xtl::span<double> _m, _b;

  std::shared_ptr<const common::IndexMap> index_map;
  int bs;
};
