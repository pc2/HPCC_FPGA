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
 | Transpose_VENDOR   | Builds the host application                    |
 | Transpose_test_VENDOR    | Compile the tests and its dependencies  |

 `VENDOR` can be `intel` or `xilinx`.
 
 More over the are additional targets to generate kernel reports and bitstreams.
 They are generated for every kernel code in the `src/device` folder:
 
  |  Target  | Description                                    |
  | -------- | ---------------------------------------------- |
  | KERNEL_FILE_NAME_`VENDOR`          | Synthesizes the kernel (takes several hours!)  |
  | KERNEL_FILE_NAME_report_`VENDOR`  | Just compile kernel and create logs and reports   |
  | KERNEL_FILE_NAME_emulate_`VENDOR`  | Create a n emulation kernel             |
  
The currently supported values for KERNEL_FILE_NAME are listed below where `transpose_diagonal` is set to be the default for the ase run:

- transpose_diagonal
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make Transpose_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
 `DATA_TYPE`     | float       | Data type used for calculation       |
`READ_KERNEL_NAME`    | transpose_read | Name of the kernel that reads A from global memory and sends it to an external channel (only needed for own implementations) |
`WRITE_KERNEL_NAME`    | transpose_write | Name of the kernel that receives a from an external channel and adds it to B (only needed for own implementations) |
`BLOCK_SIZE`     | 512          | Block size used by the kernel to transpose the matrix |
`CHANNEL_WIDTH`  | 8        | Unrolling factor for the global memory access |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

For the execution of the benchmark run:

    ./Transpose_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    $./Transpose_xilinx -h
    -------------------------------------------------------------
    General setup:
    C++ high resolution clock is used.
    The clock precision seems to be 1.00000e+01ns
    -------------------------------------------------------------
    Implementation of the matrix transposition benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 1.3

    Usage:
    ./Transpose_xilinx [OPTION...]

    -f, --file arg         Kernel file name
    -n, arg                Number of repetitions (default: 10)
    -i,                    Use memory Interleaving
        --skip-validation  Skip the validation of the output data. This will
                            speed up execution and helps when working with special
                            data types.
        --device arg       Index of the device that has to be used. If not
                            given you will be asked which device to use if there are
                            multiple devices available. (default: -1)
        --platform arg     Index of the platform that has to be used. If not
                            given you will be asked which platform to use if there
                            are multiple platforms available. (default: -1)
    -r, arg                Number of used kernel replications (default: 4)
        --test             Only test given configuration and skip execution and
                            validation
    -h, --help             Print this help
    -m, arg                Matrix size in number of blocks in one dimension
                            (default: 8)
    -b, arg                Block size in number of values in one dimension
                            (default: 512)
        --handler arg      Specify the used data handler that distributes the
                            data over devices and memory banks (default: distext)
    

    
To execute the unit and integration tests run

    ./Transpose_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Output Interpretation

An example output from an emulation is given below:

    Maximum error: 7.62939e-06 < 1.19209e-05
    Mach. Epsilon: 1.19209e-07
    Validation Time: 4.69627e+00 s
                  calc    calc FLOPS    Net [GB/s]    Mem [GB/s]
    avg:   1.15169e-01   3.72929e+10   1.49172e+11   4.47515e+11
    best:  1.14216e-01   3.76041e+10   1.50416e+11   4.51249e+11
    Validation: SUCCESS!

The output gives the average and best calculation time for the transposition with the derived metrics.
For the average and best timings, we have the following columns:

- `calc`: Calculation time in seconds, which is the pure kernel execution time without data transfer from the host.
- `calc FLOPS`: Achieved FLOPS just considering the calculation time.
- `Net [GB/s]`: Used total network bandwidth in GB/s.
- `Mem [GB/s]`: Used total global memory bandwidth in GB/s.

The `Maximum Error` field shows the largest error that was computed.
Since the arithmetic intensity of the algorithm is quite low and only one addition is required to calculate one value of the result matrix, the error should be close to the machine epsilon, which depends on the chosen data type.
The second number in the row is the validation threshold which is depending on the machine epsilon.
The validation is successful, if the maximum error is below this threshold.
The machine epsilon is given in the row below with `Mach. Epsilon`.
Moreover, the total time that was needed for the validation of the result is given, which is just a debug information.
The very last column summarizes the result: The last row will show `Validation: SUCCESS!` if the validation succeeded and the error is below the tolerated threshold.

