# GEMM Benchmark for FPGA

This repository contains the GEMM Benchmark for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

It is a modified implementation of the
[GEMM Benchmark](http://www.netlib.org/parkbench/html/matrix-kernels.html)
provided in the [HPC Challenge Benchmark](https://icl.utk.edu/hpcc/) suite.
The implementation follows the Python reference implementation given in  
_Introduction to the HPCChallenge Benchmark Suite_ available
[here](http://icl.cs.utk.edu/news_pub/submissions/hpcc-challenge-intro.pdf).

## Additional Dependencies

The benchmark *optionally* depends on a library implementing the BLAS linear-algebra interface like:

- OpenBLAS
- Intel MKL

If available, the benchmark will use `sgemm_` to validate the calculation instead of a slow reference implementation.
For matrix sizes above 1000x1000 we recommend using such a library to speed up the benchmark execution. 
Using such a library will not change the performance result of the benchmark but might affect the reported error of the calculation.

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels, where `VENDOR` can be
`intel` or `xilinx`:

 |  Target  | Description                                    |
 | -------- | ---------------------------------------------- |
 | GEMM_`VENDOR`   | Builds the host application                    |
 | Test_`VENDOR`    | Compile the tests and its dependencies  |
 
 More over the are additional targets to generate kernel reports and bitstreams.
 They are generated for every kernel code in the `src/device` folder:
 
  |  Target  | Description                                    |
  | -------- | ---------------------------------------------- |
  | gemm_cannon_`VENDOR`         | Synthesizes the kernel (takes several hours!)  |
  | gemm_cannon_report_intel   | Create an HTML report for the kernel    |
  | gemm_cannon_compile_xilinx  | Just compile kernel and create reports    |
  | gemm_cannon_emulate_`VENDOR`  | Create a n emulation kernel             |
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make fgemm

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
 `DATA_TYPE`     | float       | Data type used for calculation       |
`DEFAULT_MATRIX_SIZE` | 4096      | The default size of the quadratic matrices |
`KERNEL_NAME`| gemm | Name of the kernel (only needed for own implementations) |
`BLOCK_SIZE`    | 512          | Block size used by the kernel to transpose the matrix |
`GEMM_SIZE`    | 8             | Block size of the fully unrolled cannon block if cannon kernel is used |
`GLOBAL_MEM_UNROLL`| 16        | Unrolling factor for the global memory access |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

For execution of the benchmark run:

    ./GEMM_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./GEMM_intel -h
    
    Implementation of the GEMM benchmark proposed in the HPCC benchmark adapted for FPGA
    Usage:
      ./GEMM_intel [OPTION...]
    
      -f, --file arg        Kernel file name
      -n, arg               Number of repetitions (default: 10)
      -m, arg               Matrix size (default: 4096)
          --kernel arg      Name of the kernel (default: gemm)
      -i, --nointerleaving  Disable memory interleaving
          --device arg      Index of the device that has to be used. If not given
                            you will be asked which device to use if there are
                            multiple devices available. (default: -1)
          --platform arg    Index of the platform that has to be used. If not
                            given you will be asked which platform to use if there
                            are multiple platforms available. (default: -1)
      -h, --help            Print this help
    
To execute the unit and integration tests run

    ./Test_intel
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Output Interpretation

An example output from an emulation is given below:

    norm. resid        resid       machep
    1.45417e-05  4.76837e-05  1.19209e-07
           best         mean       GFLOPS
    6.89168e-03  6.89168e-03  1.03868e+02

The first two rows give information about the calculation error.

- `norm. resid`: The normalized residual error based on the used matrix size and used values
- `resid`: The maximum residual error of the calculation
- `machep`: The machine epsilon

The last two columns contain the time measurements and based on that the achieved FLOPS
of the calculation.

- `best`: The shortest execution time in all runs
- `mean`: Arithmetic mean of all execution times
- `GFLOPS`: GFLOPS calculated from the shortest execution time
