# HPCC FPGA

[![GitHub license](https://img.shields.io/github/license/pc2/HPCC_FPGA.svg)](https://github.com/pc2/HPCC_FPGA/blob/master/LICENSE)
[![arXiv](http://img.shields.io/badge/cs.DC-arXiv%3A2004.11059-4FC1F2.svg)](https://arxiv.org/abs/2004.11059)
[![GitHub release](https://img.shields.io/github/release/pc2/HPCC_FPGA.svg)](https://GitHub.com/pc2/HPCC_FPGA/releases/)

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

- CMake >= 3.1
- C++ compiler with C++11 support
- Intel OpenCL FPGA SDK or Xilinx Vitis
- Python 3 (for the evaluation scripts)

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

#### Build and Test Example: STREAM

As an example to configure and build the kernels of the STREAM benchmark you can follow the steps below.
The steps are very similar for all benchmarks of the suite.

Create a build directory to store the build configuration and intermediate files:
```bash
mkdir -p build/build-stream
cd build/build-stream
``` 

Configure the build using CMake and set the STREAM specific configuration options to match the target FPGA:
```bash
cmake ../../STREAM -DDEVICE_BUFFER_SIZE=8192 -DFPGA_BOARD_NAME=p520_hpc_sg280l \
    -DUSE_SVM=No -DNUM_REPLICATIONS=4
``` 

The created build configuration can then be used to build and execute the tests and create a report for the OpenCL kernel:
```bash
# Build the tests, host code and emulation kernels
make all

# Create a report for the OpenCL kernel
make stream_kernels_single_report_intel

# Execute the tests
make CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 test
```
The report can be found within the build directory of the project e.g. under `bin/stream_kernels_single/reports`.

If the tests with the selected configuration succeed and the report shows no anormalies, the kernel can be synthesized with:
```bash
# Synthesize the kernel
make stream_kernels_single

# Build the host code
make STREAM_FPGA_intel 
```

All artifacts can be found in the `bin` folder located in the current build directory.

#### Basic Functionality Testing

The subfolder `scripts` contains helper scripts that are used during the build and test process or for evaluation.
When major changes are made on the code the functionality should be checked by running all tests in the suite.
To simplify this process the script `test_all.sh` can be used to build all benchmarks with the default configuration
and run all tests.

## Publications

If you are using one of the benchmarks contained in the HPCC FPGA benchmark suite consider citing us.

#### Bibtex

    @article{hpcc_fpga,
      title={Evaluating {FPGA} Accelerator Performance with a Parameterized OpenCL Adaptation of the {HPCChallenge} Benchmark Suite},
      author={Meyer, Marius and Kenter, Tobias and Plessl, Christian},
      journal={arXiv preprint arXiv:2004.11059},
      year={2020}
    }
