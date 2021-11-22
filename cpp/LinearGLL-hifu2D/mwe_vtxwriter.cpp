#include <iostream>
#include <dolfinx>

int main(){
    dolfinx::io::VTXWriter file(MPI_COMM_WORLD, "soln.pvd", {u_n});
    return 0;
}