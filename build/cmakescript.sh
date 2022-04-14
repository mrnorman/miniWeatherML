#!/bin/bash

./cmakeclean.sh

if [ $# -ne 1 ]; then
  echo "Error: must pass exactly one parameter giving the directory to build"
  exit -1
fi

if [ ! -d $1 ]; then
  echo "Error: Passed directory does not exist"
  exit -1
fi

if [ ! -f $1/CMakeLists.txt ]; then
  echo "Error: Passed directory does not contain a CMakeLists.txt file"
  exit -1
fi

cmake      \
  -DYAKL_CUDA_FLAGS="${YAKL_CUDA_FLAGS}"         \
  -DYAKL_CXX_FLAGS="${YAKL_CXX_FLAGS}"           \
  -DYAKL_SYCL_FLAGS="${YAKL_SYCL_FLAGS}"         \
  -DYAKL_OPENMP_FLAGS="${YAKL_OPENMP_FLAGS}"     \
  -DYAKL_HIP_FLAGS="${YAKL_HIP_FLAGS}"           \
  -DYAKL_F90_FLAGS="${YAKL_F90_FLAGS}"           \
  -DCMAKE_CUDA_HOST_COMPILER="mpic++"            \
  -DNCFLAGS="${NCFLAGS}"                         \
  -DYAKL_ARCH="${YAKL_ARCH}"                     \
  -DCMAKE_EXE_LINKER_FLAGS="${TORCH_INTEL_LIBS}" \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE}"           \
  $1

ln -sf $1/inputs .

