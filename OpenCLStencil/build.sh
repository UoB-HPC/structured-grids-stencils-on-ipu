#!/bin/bash -login

#module load gcc/6.1.0
#module load cmake
#module load cuda80/toolkit/8.0.44

g++ --std=c++14 -o opencl_stencil main.cpp StencilUtils.hpp lib/cxxopts/cxxopts.hpp lib/half-2.1.0/include/half.hpp lib/lodepng/lodepng.cpp lib/lodepng/lodepng.h -lOpenCL
