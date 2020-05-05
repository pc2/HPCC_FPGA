# LINPACK for FPGA

This repository contains the LINPACK for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

The implementation is currently work in progess and is not feature complete.
Read the section **Implementation Details** for more information.


## Additional Dependencies

Additional libraries are needed to build the unit test binary:

- Intel MKL

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels, where `VENDOR` can be
`intel` or `xilinx`:

 |  Target               | Description                                    |
 | --------------------- | ---------------------------------------------- |
 | LINPACK_VENDOR      | Builds the host application linking with the Intel SDK|
 | Test_VENDOR          | Compile the tests and its dependencies linking with the Intel SDK  |
 
 More over there are additional targets to generate kernel reports and bitstreams.
 The provided kernel is optimized for the Bittware 520N board equipped with Stratix 10.
 It has a high resource utilization and will most likely not fit on smaller FPGAs.

 The kernel targets are:
 
  |  Target                        | Description                                    |
  | ------------------------------ | ---------------------------------------------- |
  | lu_blocked_pvt_VENDOR                | Synthesizes the kernel (takes several hours!)  |
  | lu_blocked_pvt_report_intel          | Create an HTML report for the kernel           |
    | lu_blocked_pvt_compile_xilinx         | Just compile kernel and create reports         |
  | lu_blocked_pvt_emulate_VENDOR          | Create a n emulation kernel                    |

 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make LINPACK_intel
    
A whole emulation build can be done with

    mkdir build && cd build
    cmake ..
    make LINPACK_intel
    
This will compile the host code as well a unit test binary with emulation kernels for functionality testing.

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`DEFAULT_MATRIX_SIZE`| 1024 | Width and heigth of the input matrix |
`GLOBAL_MEM_UNROLL`| 16        | Loop unrolling factor for all loops in the device code that load or store to global memory. This will have impact on the the width of the generated LSUs. |
`REGISTER_BLOCK_LOG`| 3        | Size of the blocks that will be processed in registers (2^3=8 is the default) |
`LOCAL_MEM_BLOCK_LOG`| 5        | Size of the blocks that will be processed in local memory (2^3=8 is the default) |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

Additionally it is possible to set the used compiler and other build tools 
in the `CMakeCache.txt` located in the build directory after running cmake.

## Execution

For execution of the benchmark run:

    ./LINPACK_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./LINPACK_intel -h
    
    Implementation of the LINPACK benchmark proposed in the HPCC benchmark suite for FPGA.
    Usage:
      ./LINPACK_intel [OPTION...]
    
      -f, --file arg      Kernel file name
      -n, arg             Number of repetitions (default: 10)
      -s, arg             Size of the data arrays (default: 1024)
          --device arg    Index of the device that has to be used. If not given
                          you will be asked which device to use if there are
                          multiple devices available. (default: -1)
          --platform arg  Index of the platform that has to be used. If not given
                          you will be asked which platform to use if there are
                          multiple platforms available. (default: -1)
      -h, --help          Print this help

    
To execute the unit and integration tests for Intel devices run

    CL_CONTEXT_EMULATOR_DEVICE=1 ./Test_intel
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Implementation Details

The benchmark will measure the elapsed time to execute a kernel for performing
an LU factorization.
It will use the time to calculate the FLOP/s.
Buffer transfer is currently not measured.
The solving of the linear equations is currently done on the CPU.

The updates are done unaligned and randomly directly on the global memory.
The repository contains two different implementations:
- `blocked`: A blocked, unoptimized kernel that performs the LU factorization
   without pivoting.
- `blocked_pvt`: Blocked kernel that performs the LU factorization with pivoting
   over the whole block.

#### Work in Progress

The implementation is currently work in progress and currently only covers the
GEFA calculation on FPGA.
A rough overview of the WIP with focus on the pivoting kernel:

- Routines C1 to C3 are not optimized and C4 reduces fMax.
- Only block-wise partial pivoting is used instead of partial pivoting over
  the whole matrix. This increases the error in the calculation.
- GESL not implemented on FPGA.


## Result Interpretation

The host code will print the results of the execution to the standard output.
The result  summary looks similar to this:

    norm. resid        resid       machep       x[0]-1     x[n-1]-1
    9.40193e+00  2.87056e-04  1.19209e-07 -3.21865e-05  2.57969e-04
    best         mean         GFLOPS       error
    1.57262e-01  1.57262e-01  7.11221e-02  9.40193e+00

The first row contains data from the correctness check that is done once when
executing the benchmark:
- `resid`: The maximum residual error when multiplying the result vector with
   the matrix and subtract by the expected result.
- `norm. resid`: The normalized residual error based on `resid`.
- `machep`: machine epsilon that gives an upper bound for rounding errors due
   to the used floating point format.
- `x[0] - 1`: The first element of the result vector minus 1. It should be
   close to 0. The same holds for `x[n-1] - 1` which is the last element of the
   vector.

The second row contains the measured performance of the benchmark:
- `best`: The best measured time for executing the benchmark in seconds.
- `mean`: The arithmetic mean of all measured execution times in seconds.
- `GFLOPS`: GFLOP/s achieved for the calculation using the best measured time.
- `error`: Same as `norm. resid` to complete the performance overview.
