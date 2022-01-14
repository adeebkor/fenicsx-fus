# C++ solver

## Requirements

- xtensor-blas (required to run `_opt` code).

## Running the code

The examples directory contains examples of various problem being solve using 
the C++ solver. For instance, to run the linear_planewave2d example, do:

1. `ffcx forms.ufl`
2. `mkdir build`
3. `cd build`
4. `cmake ..`
5. `make`
6. `./linear_planewave2d`

Accordingly, the code can also be run on multiple processes. For instance,
to run the code on 8 processes, replace instruction 6. with:

6. `mpirun -n 8 ./linear_planewave2d`
