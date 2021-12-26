# Python solver

## Installation

To install the Python solver, run:

`pip3 install . --no-cache-dir`

in this directory. This will install a Python package called hifusim.

## Running the code

The examples directory contains examples of various problem being 
solve using the hifusim package. For instance, to run the linear_hifu2d.py 
example, do:

`python3 linear_hifu2d.py`

Accordingly, the code can also be run on multiple processes. For instance, 
to run the code on 8 processes, do:

`mpirun -n 8 python3 linear_hifu2d.py`
