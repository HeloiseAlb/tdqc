#!/bin/bash
FLAGS="-I/home/bolensadrien/.local/include -I/home/bolensadrien/.local/include/armadillo_bits
-I/home/bolensadrien/.local/include/QuDyn1 -lqudyn1 -lpthread -O3 -larmadillo -std=c++11 -mtune=native -msse3 -Wall"

FLAGS2="-Wall -shared -fPIC `python3 -m pybind11 --includes` "

LIBS="-L/home/bolensadrien/.local/lib64"

IGNORE="
  -Wno-ignored-attributes
  -Wno-unused-variable
  -Wno-unused-but-set-variable
  -Wno-sign-compare
  -Wno-maybe-uninitialized
  "

# module load hdf5-1.10.5-gcc-8.2.0
# module load mkl
# module load gcc-5.2.0
# module load gcc-9.2.0


module list
which gcc-8

gcc-8 $FLAGS $FLAGS2 $LIBS $IGNORE system_pybind.cpp system.cpp hamiltonians.cpp -o system_cpp`python3-config --extension-suffix` 
# gcc $FLAGS $FLAGS2 $LIBS $IGNORE *.cpp -o system_cpp`python3-config --extension-suffix` 
