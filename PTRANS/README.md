# Matrix Transposition Benchmark for FPGA

This repository contains the Matrix Transposition Benchmark for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

It is a modified implementation of the
[Matrix Transposition Benchmark](http://www.netlib.org/parkbench/html/matrix-kernels.html)
provided in the [HPC Challenge Benchmark](https://icl.utk.edu/hpcc/) suite.
The implementation follows the Python reference implementation given in  
_Introduction to the HPCChallenge Benchmark Suite_ available
[here](http://icl.cs.utk.edu/news_pub/submissions/hpcc-challenge-intro.pdf).

## Dependencies

The benchmark comes with the following requirements for building and running:

- CMake 2.8
- GCC 4.9
- Intel OpenCL FPGA SDK 19.3

It also contains submodules that will be automatically updated when running cmake:

- cxxopts: A header only library to parse command line parameters
- googletest: A C++ test framework

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels:

 |  Target  | Description                                    |
 | -------- | ---------------------------------------------- |
 | fTrans   | Builds the host application                    |
 | Google_Tests_run| Compile the tests and its dependencies  |
 
 More over the are additional targets to generate kernel reports and bitstreams.
 They are generated for every kernel code in the `src/device` folder:
 
  |  Target  | Description                                    |
  | -------- | ---------------------------------------------- |
  | KERNEL_FILE_NAME          | Synthesizes the kernel (takes several hours!)  |
  | KERNEL_FILE_NAME_report   | Create an HTML report for the kernel    |
  | KERNEL_FILE_NAME_emulate  | Create a n emulation kernel             |
  
The currently supported values for KERNEL_FILE_NAME are:

- transpose_default
- transpose_optimized
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make fTrans

You will find all executables and kernel files in the `bin`
folder of your build directory.
You should always specify a target with make to reduce the build time!
You might want to specify predefined parameters before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
 `DATA_TYPE`     | float       | Data type used for calculation       |
`DEFAULT_DEVICE` | -1          | Index of the default device (-1 = ask) |
`DEFAULT_PLATFORM`| -1          | Index of the default platform (-1 = ask) |
`DEFAULT_REPETITIONS`| 10          | Number of times the kernel will be executed |
`KERNEL_NAME`| transpose | Name of the kernel (only needed for own implementations) |
`FPGA_BOARD_NAME`| p520_hpc_sg280l | Name of the target board |
`BLOCK_SIZE`    | 512          | Block size used by the kernel to transpose the matrix |
`GLOBAL_MEM_UNROLL`| 16        | Unrolling factor for the global memory access |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

Additionally it is possible to set the used compiler and other build tools 
in the `CMakeCache.txt` located in the build directory after running cmake.


## Execution

For execution of the benchmark run:

    ./fTrans -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./fTrans -h
    
To execute the unit and integration tests run

    ./Google_Tests_run
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Output Interpretation

An example output from an emulation is given below:

                 trans          calc    calc FLOPS   total FLOPS
    avg:   3.92349e-02   3.55022e-01   4.72568e+07   4.25540e+07
    best:  1.55398e-02   2.94127e-01   5.70407e+07   5.41782e+07
    Aggregated Error: 0.00000e00

The output gives the average time for calculation and data transfer
over all iterations as well as the best measured performance.
For both rows there are the following columns:

- `trans`: Transfer time for reading and writing buffers from host to device.
- `calc`: Actual execution time of the kernel.
- `calc FLOPS`: Achieved FLOPS just considering the execution time.
- `total FLOPS`: Total FLOPS combining transfer and calculation time.

The `Aggregated Error` field sums up the values of the result matrix. All values
should be 0 due to the used input data generation scheme.

