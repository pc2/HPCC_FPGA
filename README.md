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
- Python 3 for code generation and with [pandas](https://pandas.pydata.org) installed for the evaluation scripts

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

#### Configuration of a Benchmark

The **configuration options** are implemented as CMake build parameters and can be set when creating a new CMake build directory.
We recommend to create a new build directory for a benchmark in a folder `build` in the root direcotry of the project.
You may want to create a folder hierarchy in there e.g. to build the STREAM benchmark create a folder `build/STREAM` and change into that new folder.
Initialize a new CMake build direcotry by calling

    cmake PATH_TO_SOURCE_DIR

where `PATH_TO_SOURCE_DIR` would be `../../STREAM` in case of stream (the relative path to the source direcotry of the target benchmark).
Some of the configuration options are the same for each benchmark and are given in the Table below. 
Especially the `FPGA_BOARD_NAME` is important to set, since it will specify the target board.
The `DEFAULT_*` options are used by the host code and can also be changed later at runtime.
The given default values will be set if no other values are given during configuration.


Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`DEFAULT_DEVICE` | -1          | Index of the default device (-1 = ask) |
`DEFAULT_PLATFORM`| -1          | Index of the default platform (-1 = ask) |
`DEFAULT_REPETITIONS`| 10          | Number of times the kernel will be executed |
`FPGA_BOARD_NAME`| p520_hpc_sg280l | Name of the target board |

Additionally the compile options for the Intel or Xilinx compiler have to be specified. 
For the Intel compiler these are:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`AOC_FLAGS`| `-fpc -fp-relaxed -no-interleaving=default` | Additional Intel AOC compiler flags that are used for kernel compilation |

For the Xilinx compiler it is also necessary to set settings files for the compile and link step of the compiler.
The available options are given in the following table:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`XILINX_COMPILE_FLAGS` | `-j 40` | Set special compiler flags like the number of used threads for compilation. |
`XILINX_COMPILE_SETTINGS_FILE` | First `settings.compile.xilinx.*.ini` file found in the `settings` folder of the benchmark | Path to the file containing compile time settings like the target clock frequuency |
`XILINX_LINK_SETTINGS_FILE` | First `settings.link.xilinx.*.ini` file found in the `settings` folder of the benchmark | Path to the file containing link settings like the mapping of the memory banks to the kernel parameters |
`XILINX_GENERATE_LINK_SETTINGS` | `Yes` if the link settings file ends on `.generator.ini`, `No` otherwise | Boolean flag indicating if the link settings file will be used as a source to generate a link settings file e.g. for a given number of kernel replications |

For an overview of the current limitations of the benchmarks with regards to the Xilinx Vitis toolchain refer to the subsection [Notes on Xilinx Vitis Compatibility](#notes-on-xilinx-vitis-compatibility).

For the other benchmarks, the Xilinx configuration options will have no effect.
When building a benchmark for Xilinx FPGAs double check the path to the settings files and if they match to the target board.
The settings files follow the name convention:

    settings.[compile|link].xilinx.KERNEL_NAME.[hbm|ddr](?.generator).ini

where `KERNEL_NAME` is the name of the target OpenCL kernel file.
`hbm` or `ddr` is the type of used global memory.

All the given options can be given to CMake over the `-D` flag.

    cmake ../../RandomAccess -DFPGA_BOARD_NAME=my_board -D...

or after configuration using the UI with

    ccmake ../../RandomAccess

In the following the configuration and build steps are shown with a more specific example.

#### Build and Test Example: STREAM for Intel OpenCL FPGA SDK and the Nallatech 520N

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
In this example, `DEVICE_BUFFER_SIZE`, `USE_SVM` and `NUM_REPLICATIONS` are configuration options specific to STREAM.
Additional options can be found in the [README](STREAM/README.md) for every benchmark.


The created build configuration can then be used to build and execute the tests and create a report for the OpenCL kernel:
```bash
# Build the tests, host code and emulation kernels
make all

# Execute the tests
make CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 test

# Create a report for the OpenCL kernel
make stream_kernels_single_report_intel
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


## Notes on Xilinx Vitis Compatibility

Currently not all benchmarks fully support the Xilinx Vitis toolchain.
In this section the limitations of the benchmarks with regards to Xilinx Vitis are listed to give an overview:

- **Full Support:** STREAM, RandomAccess
- **Not optimized:** PTRANS, LINPACK, GEMM, FFT
- **No or limited testing and emulation:** LINPACK, FFT
- **Not compatible:** b_eff

*Full Support* means that the benchmarks work for the toolchain as expected.
*Not optimized* inidcates, that build, emulation and testing works, but the kernels are not optimized for the toolchain which might lead to poor performnce.
*No or limited testing and emulation* means that there are problems with the creation of the emulation kernels.
Thus the tests can not be executed for the benchmarks.
Compilation and synthesis of the kernels works for these benchmarks.
The benchmarks listed under *Not compatible* are not compatible with the Xilinx toolchain.


## Publications

If you are using one of the benchmarks contained in the HPCC FPGA benchmark suite consider citing us.

#### Bibtex

    @article{hpcc_fpga,
      title={Evaluating {FPGA} Accelerator Performance with a Parameterized OpenCL Adaptation of the {HPCChallenge} Benchmark Suite},
      author={Meyer, Marius and Kenter, Tobias and Plessl, Christian},
      journal={arXiv preprint arXiv:2004.11059},
      year={2020}
    }
