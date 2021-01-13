# Random Access Benchmark for FPGA

This repository contains the Random Access Benchmark for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

It is a modified implementation of the
[Random Access Benchmark](https://icl.utk.edu/projectsfiles/hpcc/RandomAccess/)
provided in the [HPC Challenge Benchmark](https://icl.utk.edu/hpcc/) suite.
The implementation follows the Python reference implementation given in  
_Introduction to the HPCChallenge Benchmark Suite_ available
[here](http://icl.cs.utk.edu/news_pub/submissions/hpcc-challenge-intro.pdf).

## Build

The Makefile will generate the device code using the code generator given in a submodule.
So to make use of the code generation make sure to check out the repository recursively.

The code has the following dependencies:

- C++ compiler with C++11 support
- Intel FPGA SDK for OpenCL or Xilinx Vitis
- CMake 3.1 for building

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels:

 |  Target               | Description                                    |
 | --------------------- | ---------------------------------------------- |
 | RandomAccess_`VENDOR`     | Builds the host application linking with the Intel SDK|
 | RandomAccess_test_`VENDOR`            | Compile the tests and its dependencies linking with the Intel SDK  |
 
 More over there are additional targets to generate kernel reports and bitstreams.
 The kernel targets are:
 
  |  Target                        | Description                                    |
  | ------------------------------ | ---------------------------------------------- |
  | random_access_kernels_single_`VENDOR`                | Synthesizes the kernel (takes several hours!)  |
  | random_access_kernels_single_report_`VENDOR`         | Just compile kernel and create logs and reports |
  | random_access_kernels_single_emulate_`VENDOR`          | Create a n emulation kernel                    |
  
For the host code as well as the kernels `VENDOR` can be `intel` or `xilinx`.
The report target for Xilinx is missing but reports will be generated when the kernel is synthesized.

 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make RandomAccess_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`DEFAULT_ARRAY_LENGTH`| 536870912 | Length of each input array (4GB) |
`NUM_REPLICATIONS`| 4        | Replicates the kernels the given number of times |
`DEVICE_BUFFER_SIZE`| 1       | Number of values that are stored in the local memory in the single kernel approach |
`INTEL_USE_PRAGMA_IVDEP`| No       | Use the ivdep pragma in the main loop to remove the data dependency between reads and writes. This might lead to an error larger than 1%, but might also increase performance! |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

Additionally it is possible to set the used compiler and other build tools 
in the `CMakeCache.txt` located in the build directory after running cmake.

## Execution

For execution of the benchmark run:

    ./RandomAccess_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./RandomAccess_intel -h
    
To execute the unit and integration tests for Intel devices run

    CL_CONTEXT_EMULATOR_DEVICE=1 ./RandomAccess_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Result Interpretation

The host code will print the results of the execution to the standard output.
The result  summary looks similar to this:

    Error: 9.87137e-03%
    best         mean         GUPS      
    1.73506e+01  1.73507e+01  2.47540e-01 

- `best` and `mean` are the fastest and the mean kernel execution time.
    The pure kernel execution time is measured without transferring the buffer
    back and forth to the host.
- `GUPS` contains the calculated metric _Giga Updates per Second_. It takes the
    fastest kernel execution time. The formula is
    ![GOPs = 4 * GLOBAL_MEM_SIZE / (best_time * 10^9)](https://latex.codecogs.com/gif.latex?\inline&space;GUPS&space;=&space;&bsol;frac{4&space;*&space;GLOBAL\\_MEM\\_SIZE}{&space;best\\_time&space;*&space;10^9}).
- `Error` contains the percentage of memory positions with wrong values
    after the updates where made. The maximal allowed error rate of the
    random access benchmark is 1% according to the rules given in the HPCChallenge
    specification.

Benchmark results can be found in the `results` folder in this
repository.
