# Instructions for compilation

## Prequisites
- The spectral stiffness operator requires the Fastor library. 
This can be download from this [repository](https://github.com/romeric/Fastor).
- Download the repository into the cpp/ folder.

## Example of the compilation process
1. cd mass_3d/
2. ffcx form.py
3. mkdir build
4. cd build
5. cmake ..
6. make
7. ./mass_kernel_3d
