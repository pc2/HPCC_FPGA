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
- CMake 3.15 for building

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
`DEFAULT_ARRAY_LENGTH`| 29 | Length of each input array in log2 (4GB) |
`HPCC_FPGA_RA_NUM_REPLICATIONS`| 4        | Replicates the kernels the given number of times |
`HPCC_FPGA_RA_DEVICE_BUFFER_SIZE_LOG`| 0       | Number of values that are stored in the local memory in the single kernel approach |
`HPCC_FPGA_RA_INTEL_USE_PRAGMA_IVDEP`| No       | Use the ivdep pragma in the main loop to remove the data dependency between reads and writes. This might lead to an error larger than 1%, but might also increase performance! |
`HPCC_FPGA_RA_RNG_COUNT_LOG`| 5      | Log2 of the number of random number generators that will be used concurrently |
`HPCC_FPGA_RA_RNG_DISTANCE`| 5       | Distance between RNGs in shift register. Used to relax data dependencies and increase clock frequency |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

Additionally it is possible to set the used compiler and other build tools 
in the `CMakeCache.txt` located in the build directory after running cmake.

## Execution

For execution of the benchmark run:

    ./RandomAccess_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./RandomAccess_intel -h
    
    Implementation of the random access benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 2.5

    MPI Version:  3.1
    Config. Time: Thu Dec 08 10:42:40 UTC 2022
    Git Commit:   86e0064-dirty

    Usage:
      ./bin/RandomAccess_intel [OPTION...]

      -f, --file arg          Kernel file name
      -n, arg                 Number of repetitions (default: 10)
      -i,                     Use memory Interleaving
          --skip-validation   Skip the validation of the output data. This will
                              speed up execution and helps when working with
                              special data types.
          --device arg        Index of the device that has to be used. If not
                              given you will be asked which device to use if there
                              are multiple devices available. (default: 0)
          --platform arg      Index of the platform that has to be used. If not
                              given you will be asked which platform to use if
                              there are multiple platforms available. (default: 0)
          --platform_str arg  Name of the platform that has to be used (default:
                              )
      -r, arg                 Number of used kernel replications (default: 4)
          --dump-json arg     dump benchmark configuration and results to this
                              file in json format (default: )
          --test              Only test given configuration and skip execution
                              and validation
      -h, --help              Print this help
      -d, arg                 Log2 of the size of the data array (default: 29)
      -g, arg                 Log2 of the number of random number generators
                              (default: 5)

To execute the unit and integration tests for Intel devices run

    CL_CONTEXT_EMULATOR_DEVICE=1 ./RandomAccess_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Result Interpretation

The host code will print the results of the execution to the standard output.
The result  summary looks similar to this:

    Error: 3.90625e-03

    best                mean                GUOPS
    5.04258e-04 s       7.85656e-04 s       2.03071e-03 GUOP/s

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

The json output looks like the following.

```json

{
  "config_time": "Thu Dec 08 10:42:40 UTC 2022",
  "device": "Intel(R) FPGA Emulation Device",
  "environment": {
    "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
  },
  "errors": {
    "ratio": {
      "unit": "",
      "value": 0.00390625
    }
  },
  "git_commit": "86e0064-dirty",
  "mpi": {
    "subversion": 1,
    "version": 3
  },
  "name": "random access",
  "results": {
    "guops": {
      "unit": "GUOP/s",
      "value": 0.0022880227372259515
    },
    "t_mean": {
      "unit": "s",
      "value": 0.0005729401999999999
    },
    "t_min": {
      "unit": "s",
      "value": 0.000447548
    }
  },
  "settings": {
    "#RNGs": 32,
    "Array Size": 256,
    "Communication Type": "UNSUPPORTED",
    "Kernel File": "./bin/random_access_kernels_single_emulate.aocx",
    "Kernel Replications": 4,
    "MPI Ranks": 1,
    "Repetitions": 10,
    "Test Mode": "No"
  },
  "timings": {
    "execution": [
      {
        "unit": "s",
        "value": 0.000672612
      },
      {
        "unit": "s",
        "value": 0.00058854
      },
      {
        "unit": "s",
        "value": 0.00058064
      },
      {
        "unit": "s",
        "value": 0.00057064
      },
      {
        "unit": "s",
        "value": 0.00053845
      },
      {
        "unit": "s",
        "value": 0.00055827
      },
      {
        "unit": "s",
        "value": 0.00056768
      },
      {
        "unit": "s",
        "value": 0.000649792
      },
      {
        "unit": "s",
        "value": 0.00055523
      },
      {
        "unit": "s",
        "value": 0.000447548
      }
    ]
  },
  "version": "2.5"
}

```
