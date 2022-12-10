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
    Version: 2.6

    MPI Version:  3.1
    Config. Time: Thu Mar 31 11:19:26 UTC 2022
    Git Commit:   2c3b1c6

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
                            multiple devices available. (default: -1)
        --platform arg     Index of the platform that has to be used. If not
                            given you will be asked which platform to use if there
                            are multiple platforms available. (default: -1)
    -r, arg                Number of used kernel replications (default: 5)
        --comm-type arg    Used communication type for inter-FPGA communication
                            (default: AUTO)
        --test             Only test given configuration and skip execution and
                            validation
    -h, --help             Print this help
    -m, arg                Global matrix size in number of blocks in one
                            dimension. Local matrix sizes will be determined by PQ
                            grid. (default: 1024)
    -b, arg                Log2 of the block size in number of values in one
                            dimension (default: 9)
    -p, arg                Width of the FPGA grid. The heigth (Q) will be
                            calculated from mpi_size / P. (default: 1)
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

     norm. residual      res. error          mach. eps
    4.35451e-03         5.96046e-07         1.19209e-07

     Method              best                mean                GFLOPS             
     total              1.12152e-01 s       1.16113e-01 s       2.13045e-04 GFLOP/s 
     GEFA               1.12152e-01 s       1.16113e-01 s       1.94784e-04 GFLOP/s 
     GESL               2.00000e-08 s       3.97000e-08 s       1.02400e+02 GFLOP/s 

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

The json output looks like the following.

```json

{
  "config_time": "Thu Dec 08 10:41:13 UTC 2022",
  "device": "Intel(R) FPGA Emulation Device",
  "environment": {
    "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
  },
  "errors": {
    "epsilon": {
      "unit": "",
      "value": 1.1920928955078125e-07
    },
    "residual": {
      "unit": "",
      "value": 5.960464477539062e-07
    },
    "residual_norm": {
      "unit": "",
      "value": 0.004354506590071576
    }
  },
  "git_commit": "86e0064-dirty",
  "mpi": {
    "subversion": 1,
    "version": 3
  },
  "name": "LINPACK",
  "results": {
    "gflops": {
      "unit": "GFLOP/s",
      "value": 0.000213044786995575
    },
    "gflops_lu": {
      "unit": "GFLOP/s",
      "value": 0.00019478383998887983
    },
    "gflops_sl": {
      "unit": "GFLOP/s",
      "value": 102.4
    },
    "t_mean": {
      "unit": "s",
      "value": 0.1161132923
    },
    "t_min": {
      "unit": "s",
      "value": 0.112151692
    },
    "tlu_mean": {
      "unit": "s",
      "value": 0.11611325259999998
    },
    "tlu_min": {
      "unit": "s",
      "value": 0.112151672
    },
    "tsl_mean": {
      "unit": "s",
      "value": 3.97e-08
    },
    "tsl_min": {
      "unit": "s",
      "value": 2e-08
    }
  },
  "settings": {
    "Block Size": 16,
    "Communication Type": "IEC",
    "Data Type": "cl_float",
    "Emulate": false,
    "FPGA Torus": {
      "P": 1,
      "Q": 1
    },
    "Kernel File": "./bin/hpl_torus_IEC_emulate.aocx",
    "Kernel Replications": 3,
    "MPI Ranks": 1,
    "Matrix Size": 32,
    "Repetitions": 10,
    "Test Mode": "No"
  },
  "timings": {
    "gefa": [
      {
        "unit": "s",
        "value": 0.112151672
      },
      {
        "unit": "s",
        "value": 0.112186842
      },
      {
        "unit": "s",
        "value": 0.114559183
      },
      {
        "unit": "s",
        "value": 0.114920089
      },
      {
        "unit": "s",
        "value": 0.113395783
      },
      {
        "unit": "s",
        "value": 0.113512676
      },
      {
        "unit": "s",
        "value": 0.118974459
      },
      {
        "unit": "s",
        "value": 0.11378015
      },
      {
        "unit": "s",
        "value": 0.131815478
      },
      {
        "unit": "s",
        "value": 0.115836194
      }
    ],
    "gesl": [
      {
        "unit": "s",
        "value": 2e-08
      },
      {
        "unit": "s",
        "value": 3e-08
      },
      {
        "unit": "s",
        "value": 3e-08
      },
      {
        "unit": "s",
        "value": 2.9e-08
      },
      {
        "unit": "s",
        "value": 1.5e-07
      },
      {
        "unit": "s",
        "value": 3e-08
      },
      {
        "unit": "s",
        "value": 2e-08
      },
      {
        "unit": "s",
        "value": 2.9e-08
      },
      {
        "unit": "s",
        "value": 2.9e-08
      },
      {
        "unit": "s",
        "value": 3e-08
      }
    ]
  },
  "version": "2.6"
}

```
