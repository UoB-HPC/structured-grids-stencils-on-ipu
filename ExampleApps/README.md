# Example applications

The example applications stencil discussed in our paper are 
(1) A Gaussian Blur image filter, and
(2) A 2D Lattice Boltzmann fluid simulation

## Halo Region Approaches
Weve also included a program and associated codelets that demonstrate various approaches to performing halo exchange using the Poplar framework. We demonstrate an ''implicit' approach, in which we assign slices of a large tensor to different tile memories, and also use the Poplar functions for appending and concatenating tensors to build a description of the data with associated halo regions. The compiler then figures out the necessary communication. We also demonstrate an 'explicit' approach in which extra padding regions are left in the large grid tensor and halos are manually copied using `Copy` programs. Note that the order in which these operations are grouped drastically affects halo exchange performance. Running the explicit approach on the IPU is only possible with SDK >1.4, because of a compiler issue we identified that prevented the same tensor being copied in two different directions in the same compute set. 


If you're interested in efficiently implementing halo exchange, we note that our latest work uses a different strategy altogether (we changed our strategy subsequent to the publication of our paper. 
In short, we now assign each tile a slice of a large `char[]` tensor, and give every Vertex `InOut<Vector<char, ...>>` access to this slice, but cast it to our desired data type as the first step in each vertex, and manage our memory separately. This allows us to achieve good data alignment for vectorisatin without incurring the data rearrangement costs discussed in the paper.

## Gaussian Blur

We implement the Gaussian Blur 2 ways for the IPU: 
(1) using the `popops` library to work on Tensors directly, similar to 
what an approach using NumPy would take
(2) implemented as a stencil application in lower level Poplar

For comparison, an OpenCL version is in the top-level directory [(here)](../OpenCLStencil).

The Gaussian Blur code in `main` includes a naive CPU implementation in standard C++ (`GaussianBlurCpu.hpp`). 
This version is only for illustration - the CPU results we discussed in our paper were run against the OpenCL version. Use this version to familiarise yourself with what a simple solution looks like before looking at the IPU versions.

The Poplibs/Popops version in `GaussianBlurPoplibs.cpp` demonstrates how to solve this problem using Poplar')s convolution functions. It needs no custom vertices (codelets).

The lower-level version in `GaussianBlurLowLevel.cpp` demonstrates how to solve this problem in low level Poplar, with the associated custom vertexes in `codelets/GaussinBlurCodelets.cpp`. We also show a vectorised, optimised version, which we used for our timings. These use the vectorised data types (e.g. `float2`), which are currently poorly documented in Graphcore's Poplar documentation. We recommend looking at the headers and Poplibs source to understand more about how custom
vertex code can benefit from these.


## Lattice-Boltzmann Simulation
The OpenCL versions we used for CPU and GPU comparisons are available from the authors on request. We haven't published them, since they form the solutions to an exercise in the University of Bristol's HPC course, but are happy to share with interested researchers and those wishing to replicate our results.

The Low-level Poplar versions are in `LbmAos.cpp` with the associated  `codelets/D2Q9Codelets.cpp` custom vertexes.

## Common utility code
Common code for loading images, writing results, partitioning and mapping tensors etc. is in the `include` directory. Note that Poputils now includes standard functions to do many of these tasks. We implemented our own to experiment with the impact of different 2D partitioning strategies, but recommend using the Poplar-provided functions where possible.

## Building the code
If you have access to an IPU system, make sure the Poplar SDK is installed and your paths have been set up as described in the Graphcore documentation, to allow finding the Poplar libraries and headers.

Using cmake, you can build the targets described in `CMakeLists.txt`.

