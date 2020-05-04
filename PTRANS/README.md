# Matrix Transposition Benchmark for FPGA

This repository contains the Matrix Transposition Benchmark for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

It is a modified implementation of the
[Matrix Transposition Benchmark](http://www.netlib.org/parkbench/html/matrix-kernels.html)
provided in the [HPC Challenge Benchmark](https://icl.utk.edu/hpcc/) suite.
The implementation follows the Python reference implementation given in  
_Introduction to the HPCChallenge Benchmark Suite_ available
[here](http://icl.cs.utk.edu/news_pub/submissions/hpcc-challenge-intro.pdf).

## Additional Dependencies

The benchmark needs no additional dependencies than the ones given in the main [README](../README.md).

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels:

 |  Target  | Description                                    |
 | -------- | ---------------------------------------------- |
 | trans_VENDOR   | Builds the host application                    |
 | Test_VENDOR    | Compile the tests and its dependencies  |

 `VENDOR` can be `intel` or `xilinx`.
 
 More over the are additional targets to generate kernel reports and bitstreams.
 They are generated for every kernel code in the `src/device` folder:
 
  |  Target  | Description                                    |
  | -------- | ---------------------------------------------- |
  | KERNEL_FILE_NAME_VENDOR          | Synthesizes the kernel (takes several hours!)  |
  | KERNEL_FILE_NAME_report_intel  | Create an HTML report for the kernel    |
  | KERNEL_FILE_NAME_emulate_VENDOR  | Create a n emulation kernel             |
| KERNEL_FILE_NAME_compile_xilinx          | Just compile kernel and create logs and reports |
  
The currently supported values for KERNEL_FILE_NAME are listed below where `transpose_optimized` is set to be the default for the ase run:

- transpose_default
- transpose_optimized
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make trans_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
 `DATA_TYPE`     | float       | Data type used for calculation       |
`KERNEL_NAME`| transpose | Name of the kernel (only needed for own implementations) |
`BLOCK_SIZE`    | 512          | Block size used by the kernel to transpose the matrix |
`GLOBAL_MEM_UNROLL`| 16        | Unrolling factor for the global memory access |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

For execution of the benchmark run:

    ./trans_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./trans_intel -h
    
    Implementation of the matrix transposition benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 0.1.2
    Usage:
      ./trans_intel [OPTION...]
    
      -f, --file arg        Kernel file name
      -n, arg               Number of repetitions (default: 10)
      -m, arg               Matrix size (default: 4096)
          --kernel arg      Name of the kernel (default: transpose)
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

