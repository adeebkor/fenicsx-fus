#! /bin/bash

rm -rf build/ a.out cmake_hdf5_test.o
mkdir build/
cmake -S . -B build/