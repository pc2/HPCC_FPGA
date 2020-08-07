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

- CMake >= 3.13
- C++ compiler with C++11 support
- Intel OpenCL FPGA SDK or Xilinx Vitis
- Python 3 for code generation and with [pandas](https://pandas.pydata.org) installed for the evaluation scripts

Moreover, additional libraries are fetched by the build system during configuration:

- [cxxopts](https://github.com/jarro2783/cxxopts) for option parsing
- [hlslib](https://github.com/definelicht/hlslib) for CMake FindPackages
- [Googletest](https://github.com/google/googletest) for unit testing

These dependencies will be downloaded automatically when configuring a benchmark for the first time.
Besides that, some benchmarks might need additional dependencies.
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
`INTEL_CODE_GENERATION_SETTINGS` | "" | Path to the settings file that will be used as input for the code generator script. It may contain additional variables or functions.  |

For the Xilinx compiler it is also necessary to set settings files for the compile and link step of the compiler.
The available options are given in the following table:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`XILINX_COMPILE_FLAGS` | `-j 40` | Set special compiler flags like the number of used threads for compilation. |
`XILINX_COMPILE_SETTINGS_FILE` | First `settings.compile.xilinx.*.ini` file found in the `settings` folder of the benchmark | Path to the file containing compile time settings like the target clock frequuency |
`XILINX_LINK_SETTINGS_FILE` | First `settings.link.xilinx.*.ini` file found in the `settings` folder of the benchmark | Path to the file containing link settings like the mapping of the memory banks to the kernel parameters |
`XILINX_GENERATE_LINK_SETTINGS` | `Yes` if the link settings file ends on `.generator.ini`, `No` otherwise | Boolean flag indicating if the link settings file will be used as a source to generate a link settings file e.g. for a given number of kernel replications |

When building a benchmark for Xilinx FPGAs double check the path to the settings files and if they match to the target board.
The settings files follow the name convention:

    settings.[compile|link].xilinx.KERNEL_NAME.[hbm|ddr](?.generator).ini

where `KERNEL_NAME` is the name of the target OpenCL kernel file.
`hbm` or `ddr` is the type of used global memory.

All the given options can be given to CMake over the `-D` flag.

    cmake ../../RandomAccess -DFPGA_BOARD_NAME=my_board -D...

or after configuration using the UI with

    ccmake ../../RandomAccess


For an overview of the current limitations of the benchmarks refer to the subsection [Notes on Vendor Vitis Compatibility](#notes-on-vendor-compatibility).
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

## Code Documentation

The benchmark suite supports the generation of code documentation using doxygen in HTML and Latex format.
To generate the documentation, execute the following commands:

    cd docs
    doxygen doxy.config

The generated documentation will be placed in `docs/html` and `docs/latex`.
To view the HTML documentation, open `docs/html/index.html` with a internet browser.

A more general documentation is maintained using Sphinx. It can be generated using the makefile provided in the `docs/` folder.
In this documentation a more general description of the benchmarks and how to use them is given.
To generate the HTML documentation, execute the commands below:

    cd docs
    make html

## Custom Kernels

The benchmark suite also allows to use custom kernels for measurements.
Some basic setup is already done for all benchmarks to ensure an easier integration of custom kernels into the build system.
Every benchmark comes with an folder `src/device/custom` that is meant to be used for customized or own kernel designs.
The folder already contains a `CMakeLists.txt` file that creates build targets for all OpenCL files (*.cl) in this folder and also creates tests for each custom kernel using CTest.
Place the custom kernel design in this folder to use it within the build system of the benchmark suite.
To enable custom kernel builds the build environment has to be configured using the `USE_CUSTOM_KERNEL_TARGETS` flag.
As an example, the following call will create a build environment that supports custom kernels for STREAM:

    mkdir build; cd build
    cmake ../STREAM -DUSE_CUSTOM_KERNEL_TARGETS=Yes

After adding a new custom kernel to the folder, rerun CMake to generate build targets and tests for this kernel.
Now the same build commands that are used for the base impementations can be used for the custom implementations.
This also means that the kernels will also be tested within the `make test` command.


## Notes on Vendor Compatibility

Current FPGA boards come with different types of memory that need specific support in the device or host code.
The most common memory types that this overview is focusing on are:

- *Shared Virtual Memory (SVM)*: The FPGA directly accesses the data on the host's memory over the PCIe connection. No copying by the host is necessary, but the memory bandwidth of the FPGA is limited by the PCIe connection.
- *DDR*: The FPGA board is equipped with one or more DDR memory banks and the host is in charge of copying the data forth and back over the PCIe connection.
This allows higher memory bandwidths during kernel execution.
- *High Bandwidth Memory (HBM)*: The FPGA fabric itself is equipped with memory banks that can be accessed by the host to copy data. Compared to DDR, this memory type consists of more, but smaller memory banks so that the host needs to split the data between all memory banks to achieve the best performance. Still, the total achievable memory bandwidth is much higher compared to DDR.

The following three tables contain an overview of the compatibility of all benchmarks that use global memory with the three mentioned memory types.
b_eff is not included since it does not use global memory.
Full support of the benchmark is indicated with a **Yes**, functionally correct behavior but performance limitations are indicated with **(Yes)**, no support is indicated with **No**.
For Xilinx, all benchmarks need a compatible compile- and link-settings-file to map the kernel memory ports to the available memory banks.

#### DDR memory

| Benchmark    | Intel      | Xilinx       |
|--------------|------------|--------------|
| STREAM       | Yes        |  Yes         |            
| RandomAccess | Yes        |  Yes         |      
| PTRANS       | Yes        |  Yes         |      
| LINPACK      | (Yes)      |  (Yes)       |           
| GEMM         | Yes        |  Yes         |      
| FFT          | Yes        |  Yes         |       


#### HBM

*(Yes)* indicates, that the benchmarks can be excuted with HBM, but not all available memory banks can be used. For Intel, the device code has to be modified to make it compatible with HBM.

| Benchmark    | Intel      | Xilinx       |
|--------------|------------|--------------|
| STREAM       | Yes        |  Yes         |            
| RandomAccess | Yes        |  Yes         |      
| PTRANS       | No         |  (Yes)       |      
| LINPACK      | No         |  (Yes)       |           
| GEMM         | No         |  Yes         |      
| FFT          | No         |  Yes         | 

#### SVM

SVM could not be tested with Xilinx-based boards, yet. Thus, they are considered as not working.

| Benchmark    | Intel      | Xilinx       |
|--------------|------------|--------------|
| STREAM       | Yes        |  No          |            
| RandomAccess | Yes        |  No          |      
| PTRANS       | Yes        |  No          |      
| LINPACK      | Yes        |  No          |           
| GEMM         | Yes        |  No          |      
| FFT          | Yes        |  No          | 

## Publications

If you are using one of the benchmarks contained in the HPCC FPGA benchmark suite consider citing us.

#### Bibtex

    @article{hpcc_fpga,
      title={Evaluating {FPGA} Accelerator Performance with a Parameterized OpenCL Adaptation of the {HPCChallenge} Benchmark Suite},
      author={Meyer, Marius and Kenter, Tobias and Plessl, Christian},
      journal={arXiv preprint arXiv:2004.11059},
      year={2020}
    }
