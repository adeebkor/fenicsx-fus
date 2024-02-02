# Python solver

## Installation

To install the Python solver, run:

`pip3 install . --no-cache-dir`

in this directory. This will install a Python package called fenicsxfus.

## Running the code

The examples directory contains examples of various problem being 
solve using the fenicsxfus package. For instance, to run the
linear_planewave2d_2.py example, do:

`python3 linear_planewave2d_2.py`

Accordingly, the code can also be run on multiple processes. For instance, 
to run the code on 8 processes, do:

`mpirun -n 8 python3 linear_planewave2d_2.py`
