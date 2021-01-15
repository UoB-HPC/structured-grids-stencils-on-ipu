cmake_minimum_required(VERSION 3.12)
project(opencl_stencil)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -ansi -Wno-deprecated")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -ansi -Wno-deprecated -march=native -mtune=native -O3")
endif()
#
#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#endif()

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_libraries(${OpenCL_LIBRARY})

include_directories(lib/lodepng)
include_directories(lib/cxxopts)
include_directories(lib/half-2.1.0/include)

add_executable(opencl_stencil main.cpp StencilUtils.hpp lib/lodepng/lodepng.cpp)
target_link_libraries(opencl_stencil ${OpenCL_LIBRARY})
