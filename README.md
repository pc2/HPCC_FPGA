# HPCC FPGA

HPCC FPGA is an OpenCL-based FPGA benchmark suite with focus on high performance computing.
It is based on the benchmarks of the well-established CPU benchmark suite [HPCC](https://icl.utk.edu/hpcc/).
This repository contains the OpenCL kernels and host code for all benchmarks together with the build scripts and instructions.

## Overview

Every benchmark comes in a separate subfolder and with its own build scripts.
This allows the individual configuration of every benchmark and also the use of different dependencies 
(e.g. the network benchmark *b_eff* might need a different BSP than the other benchmarks).

The included benchmarks are listed below.
You can find more information on how to build the benchmarks in the appropriate subfolder.

- [b_eff](b_eff): Ths application sends messages of varying sizes of the inter-FPGA network and measures the achieved bandwidth.
- [FFT](FFT): Executes multiple 1d-FFT of size 2^12.
- [GEMM](GEMM): Multiplies two quadratic matrices similar to the GEMM routine implemented in BLAS.
- [LINPACK](LINPACK): Implementation of the [LINPACK benchmark](https://www.netlib.org/benchmark/hpl/) for FPGA. (WIP and currently only does a LU factorization with block-wise pivoting)
- [PTRANS](PTRANS): Transposes a quadratic matrix.
- [RandomAccess](RandomAccess): Executes updates on a data array following a pseudo random number scheme.
- [STREAM](STREAM): Implementation of the [STREAM benchmark](https://www.cs.virginia.edu/stream/) for FPGA.

The repository contains multiple submodules located in the `extern` folder.

## General Build Setup

The build setup is very similar for all benchmarks in the suite.
Every benchmark comes with a separate CMake project with host and device code.

#### General Dependencies

All benchmarks come with the following build dependencies:

- CMake (Version 2.8/3.1 depending on the benchmark)
- C++ compiler with C++11 support
- Intel OpenCL FPGA SDK or Xilinx Vitis

Moreover the host code and the build system use additional libraries included as git submodules:

- [cxxopts](https://github.com/jarro2783/cxxopts) for option parsing
- [hlslib](https://github.com/definelicht/hlslib) for CMake FindPackages
- [Googletest](https://github.com/google/googletest) for unit testing

Make sure to initialize and update the submodules before building a benchmark with:

    git submodule update --init --recursive

After that please follow the instructions given in the README of the individual benchmark to configure and build the FPGA bitstream and host code.
Also some benchmarks might need additional dependencies.
More information on that can be found in the README located in the subfolder for each benchmark.
One key feature of all benchmarks of this suite is that they come with individual **configuration options**.
They can be used to adjust the OpenCL base implementations of a benchmark for a specific FPGA architecture and optimize the performance and resource usage.

#### Build Example: STREAM

As an example to configure and build the kernels of the STREAM benchmark you can follow the steps below.

Create a build directory to store the build configuration and intermediate files:
```    
mkdir -p build/build-stream
cd build/build-stream
``` 

Configure the build using CMake and set the STREAM specific configuration options to match the target FPGA:
```
cmake ../../STREAM -DDEVICE_BUFFER_SIZE=8192 -DFPGA_BOARD_NAME=p520_hpc_sg280l \
    -DUSE_SVM=No -DNUM_REPLICATIONS=4
``` 

The created build configuration can then be used to build and execute the tests and create a report for the OpenCL kernel:
```
make Test_intel stream_kernels_single_report_intel
cd bin
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./Test_intel 
```
The report can be found within the build directory of the project e.g. under `bin/stream_kernels_single/reports`.

If the tests with the selected configuration succeed and the report shows no anormalies, the kernel can be synthesized with:
```
# Synthesize the kernel
make stream_kernels_single

# Build the host code
make STREAM_FPGA_intel 
```

All artifacts can be found in the `bin` folder located in the current build directory.

## Publications

If you are using one of the benchmarks contained in the HPCC FPGA benchmark suite consider citing us.

#### Bibtex

    @article{hpcc_fpga,
      title={Evaluating {FPGA} Accelerator Performance with a Parameterized OpenCL Adaptation of the {HPCChallenge} Benchmark Suite},
      author={Meyer, Marius and Kenter, Tobias and Plessl, Christian},
      journal={arXiv preprint arXiv:2004.11059},
      year={2020}
    }
