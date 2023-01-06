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

For half precision support, the IEEE 754-based half-precision floating-point library by Christian Rau is used and a copy is provided with this code. 

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels, where `VENDOR` can be
`intel` or `xilinx`:

 |  Target  | Description                                    |
 | -------- | ---------------------------------------------- |
 | GEMM_`VENDOR`   | Builds the host application                    |
 | GEMM_test_`VENDOR`    | Compile the tests and its dependencies  |
 
 More over the are additional targets to generate kernel reports and bitstreams.
 They are generated for every kernel code in the `src/device` folder:
 
  |  Target  | Description                                    |
  | -------- | ---------------------------------------------- |
  | gemm_cannon_`VENDOR`         | Synthesizes the kernel (takes several hours!)  |
  | gemm_cannon_report_`VENDOR`  | Just compile kernel and create reports    |
  | gemm_cannon_emulate_`VENDOR`  | Create a n emulation kernel             |
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make GEMM_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
 `DATA_TYPE`     | float (also supported: half, double)      | Data type used for calculation. *Note: Currently, half-precision does not work on Intel FPGAs because they can not be passed as kernel argument per value.*  |
`DEFAULT_MATRIX_SIZE` | 8      | The default size of the quadratic matrices in blocks |
`BLOCK_SIZE`    | 512          | Block size used by the kernel for calculation |
`GEMM_SIZE`    | 8             | Block size of the fully unrolled matrix multiplication in registers |
`GLOBAL_MEM_UNROLL`| 16        | Unrolling factor for the global memory access |
`INTEL_MUL_SHIFT_REG`| 0       | Size of the shift register that can be optionally used by the Intel implementation to relax data dependencies (defaults to 0, which means that no shift register is used) |
`NUM_REPLICATIONS` | 4         | Number of kernel replications. Every kernel will calculate a part of the output matrix |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

For execution of the benchmark run:

    ./GEMM_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./GEMM_intel -h

    Implementation of the GEMM benchmark proposed in the HPCC benchmark adapted for FPGA
    Version: 1.3

    MPI Version:  3.1
    Config. Time: Thu Dec 08 10:39:51 UTC 2022
    Git Commit:   86e0064-dirty

    Usage:
      ./bin/GEMM_intel [OPTION...]

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
      -m, arg                 Matrix size in number of blocks in a single
                              dimension (default: 8)
      -b, arg                 Block size in number of values in one dimension
                              (default: 32)
          --replicate-inputs  Also replicates the input buffer for each kernel

To execute the unit and integration tests run

    ./GEMM_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Output Interpretation

An example output from an emulation is given below:

     norm. residual      res. error          mach. eps          
     8.08345e-05         7.62939e-06         1.19209e-07        

     best                mean                GFLOPS             
     6.50672e-03 s       1.06789e-02 s       5.15689e+00 GFLOP/s

The first two rows give information about the calculation error.

- `norm. residual`: The normalized residual error based on the used matrix size and used values
- `res. error`: The maximum residual error of the calculation
- `mach. epsilon`: The machine epsilon

The last two columns contain the time measurements and based on that the achieved FLOPS
of the calculation.

- `best`: The shortest execution time in all runs
- `mean`: Arithmetic mean of all execution times
- `GFLOPS`: GFLOPS calculated from the shortest execution time

The json output looks like the following.

```json

{
  "config_time": "Wed Dec 14 08:40:52 UTC 2022",
  "device": "Intel(R) FPGA Emulation Device",
  "environment": {
    "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
  },
  "errors": {
    "epsilon": 1.1920928955078125e-07,
    "residual": 7.62939453125e-06,
    "residual_norm": 8.08345175162664e-05
  },
  "execution_time": "Wed Dec 14 09:14:09 UTC 2022",
  "git_commit": "be1a4e9-dirty",
  "mpi": {
    "subversion": 1,
    "version": 3
  },
  "name": "GEMM",
  "results": {
    "gflops": {
      "unit": "GFLOP/s",
      "value": 5.297554068962992
    },
    "t_mean": {
      "unit": "s",
      "value": 0.010202154299999999
    },
    "t_min": {
      "unit": "s",
      "value": 0.006333948
    }
  },
  "settings": {
    "Block Size": 32,
    "Communication Type": false,
    "Kernel File": false,
    "Kernel Replications": 4,
    "MPI Ranks": 1,
    "Matrix Size": 256,
    "Repetitions": 10,
    "Replicate Inputs": false,
    "Test Mode": false
  },
  "timings": {
    "execution": [
      {
        "unit": "s",
        "value": 0.012732567
      },
      {
        "unit": "s",
        "value": 0.006511861
      },
      {
        "unit": "s",
        "value": 0.006333948
      },
      {
        "unit": "s",
        "value": 0.012710817
      },
      {
        "unit": "s",
        "value": 0.006552662
      },
      {
        "unit": "s",
        "value": 0.006600733
      },
      {
        "unit": "s",
        "value": 0.012673167
      },
      {
        "unit": "s",
        "value": 0.012720237
      },
      {
        "unit": "s",
        "value": 0.012608296
      },
      {
        "unit": "s",
        "value": 0.012577255
      }
    ]
  },
  "validated": true,
  "version": "1.3"
}

```
