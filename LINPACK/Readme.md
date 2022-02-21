# LINPACK for FPGA

This repository contains the LINPACK for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.


## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels, where `VENDOR` can be
`intel` or `xilinx`:

 |  Target               | Description                                    |
 | --------------------- | ---------------------------------------------- |
 | Linpack_`VENDOR`      | Builds the host application linking with the Intel SDK|
 | Linpack_test_`VENDOR`          | Compile the tests and its dependencies linking with the Intel SDK  |
 
 Moreover, there are additional targets to generate kernel reports and bitstreams.
 The provided kernel is optimized for the Bittware 520N board equipped with Stratix 10.
 Only the LU facotrization without pivoting is implemented on FPGA and external channels are
 used to calculate the solution in a 2D torus of FPGAs.

 The kernel targets are listed below. `COMM_TYPE` can be IEC for Intel external channel (only available for vendor Intel) and PCIE for communication via PCIe and MPI.
 
  |  Target                        | Description                                    |
  | ------------------------------ | ---------------------------------------------- |
  | hpl_torus_`COMM_TYPE`_`VENDOR`                | Synthesizes the kernel (takes several hours!)  |
  | hpl_torus_`COMM_TYPE`_report_`VENDOR`         | Just compile kernel and create reports         |
  | hpl_torus_`COMM_TYPE`_emulate_`VENDOR`          | Create a n emulation kernel                    |

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
`REGISTER_BLOCK_LOG`| 3        | Size of the blocks that will be processed in registers (2^3=8 is the default) |
`LOCAL_MEM_BLOCK_LOG`| 5        | Size of the blocks that will be processed in local memory (2^3=8 is the default) |
`DATA_TYPE`     | float        | Used data type. Can be `float` or `double` |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

Additionally it is possible to set the used compiler and other build tools 
in the `CMakeCache.txt` located in the build directory after running cmake.

## Execution

For execution of the benchmark run:

    ./Linpack_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./Linpack_intel -h
    
    Implementation of the LINPACK benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 2.3

    MPI Version:  3.1

    Usage:
    bin/Linpack_intel [OPTION...]

    -f, --file arg         Kernel file name
    -n, arg                Number of repetitions (default: 10)
    -i,                    Use memory Interleaving
        --skip-validation  Skip the validation of the output data. This will
                            speed up execution and helps when working with special
                            data types.
        --device arg       Index of the device that has to be used. If not
                            given you will be asked which device to use if there are
                            multiple devices available. (default: 0)
        --platform arg     Index of the platform that has to be used. If not
                            given you will be asked which platform to use if there
                            are multiple platforms available. (default: 0)
    -r, arg                Number of used kernel replications (default: 1)
        --comm-type arg    Used communication type for inter-FPGA communication
                            (default: AUTO)
        --test             Only test given configuration and skip execution and
                            validation
    -h, --help             Print this help
    -m, arg                Matrix size in number of blocks in one dimension for
                            a singe MPI rank. Total matrix will have size m *
                            sqrt(MPI_size) (default: 2)
    -b, arg                Log2 of the block size in number of values in one
                            dimension (default: 5)
        --uniform          Generate a uniform matrix instead of a diagonally
                            dominant. This has to be supported by the FPGA kernel!
        --emulation        Use kernel arguments for emulation. This may be
                            necessary to simulate persistent local memory on the FPGA

Available options for `--comm-type`:

- `IEC`: Intel external channels are used by the kernels for communication.
- `PCIE`: PCIe and MPI are used to exchange data between FPGAs over the CPU.
    
To execute the unit and integration tests for Intel devices run

    ./Linpack_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.


## Result Interpretation

The host code will print the results of the execution to the standard output.
The result  summary looks similar to this:

    norm. resid        resid       machep   
        3.25054e-08    5.88298e-05    1.19209e-07
    Validation Time: 4.55059e+01 s
            Method           best           mean         GFLOPS
            total    5.87510e+01    5.87510e+01    2.10546e+04
            GEFA    5.87510e+01    5.87510e+01    2.10541e+04
            GESL    4.70000e-08    4.70000e-08    6.42532e+08
    Validation: SUCCESS!

The first row contains data from the correctness check that is done once when
executing the benchmark:
- `resid`: The maximum residual error when multiplying the result vector with
   the matrix and subtract by the expected result.
- `norm. resid`: The normalized residual error based on `resid`.
- `machep`: machine epsilon that gives an upper bound for rounding errors due
   to the used floating point format.

The table below contains the performance measurements for the bechmark for the both routines GEFA and GESL.
Only GEFA is implemented on FPGA, so only this result is significant for now. 
*GESL measurement is currently disabled and does not show any valid results since execution is done on the host during validation!*
The columns of the table contain the following information:
- `best`: The best measured time for executing the benchmark in seconds.
- `mean`: The arithmetic mean of all measured execution times in seconds.
- `GFLOPS`: GFLOP/s achieved for the calculation using the best measured time.

The last row of the output will always contain `Validation: SUCCESS!`, if the norm. residual is below 1.
This will be interpreted as successful validation.
In this case, the executable will return 0 as exit code, 1 otherwise.
