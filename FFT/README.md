# FFT Benchmark for FPGA

This repository contains the FFT Benchmark for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

It is based on the FFT benchmark of the [HPC Challenge Benchmark](https://icl.utk.edu/hpcc/) suite.
The FFT1D reference implementation is used for the kernel code.

## Additional Dependencies

The benchmark needs no additional dependencies than the ones given in the main [README](../README.md).

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels, where `VENDOR` can be
`intel` or `xilinx`:

 |  Target  | Description                                    |
 | -------- | ---------------------------------------------- |
 | FFT_`VENDOR`     | Builds the host application                    |
 | FFT_test_`VENDOR`    | Compile the tests and its dependencies  |
 
 More over the are additional targets to generate kernel reports and bitstreams.
 The provided kernel is optimized for Stratix 10 with 512bit LSUs.
 The kernel targets are:
 
  |  Target  | Description                                    |
  | -------- | ---------------------------------------------- |
  | fft1d_float_8_`VENDOR`         | Synthesizes the kernel (takes several hours!)  |
  | fft1d_float_8_report_`VENDOR`   | Create a report for the kernel    |
  | fft1d_float_8_emulate_`VENDOR`  | Create a n emulation kernel             |
  
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make FFT_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`DEFAULT_ITERATIONS`| 100          | Default number of iterations that is done with a single kernel execution|
`LOG_FFT_SIZE`   | 12          | Log2 of the FFT Size that has to be used i.e. 3 leads to a FFT Size of 2^3=8|
`NUM_REPLICATIONS` | 1         | Number of kernel replications. The whole FFT batch will be divided by the number of compute kernels. |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

For execution of the benchmark run:

    ./FFT_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./FFT_intel -h
    
    Implementation of the FFT benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 1.4
    Usage:
      ./FFT_intel [OPTION...]
    
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
      -r, arg                 Number of used kernel replications (default: 1)
          --dump-json arg     dump benchmark configuration and results to this
                              file in json format (default: )
          --test              Only test given configuration and skip execution
                              and validation
      -h, --help              Print this help
      -b, arg                 Number of batched FFT calculations (iterations)
                              (default: 100)
          --inverse           If set, the inverse FFT is calculated instead
    
To execute the unit and integration tests run

    ./FFT_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Output Interpretation

The benchmark will print the following two tables to standard output after execution:

     res. error          mach. eps
     2.63523e-01         1.19209e-07

                     avg                 best
          Time in s: 8.93261e-04 s       8.73572e-04 s
             GFLOPS: 2.75127e-01 GFLOP/s 2.81328e-01 GFLOP/s

          
The first table contains the maximum residual error of the calculation and the
machine epsilon that was used to calculate the residual error.
The benchmark will perform a FFT with the FPGA kernel on random input data.
In a second step the resulting data will be used as input for an iFFT using a CPU
reference implementation in double precision.
The residual error is then calculated with:

![res=\frac{||x-x'||}{\epsilon*ld(n)}](https://latex.codecogs.com/gif.latex?res=\frac{||x-x'||}{\epsilon*ld(n)})

where `x` is the input data of the FFT, `x'` the resulting data from the iFFT, epsilon the machine epsilon and `n` the FFT size.

In the second table the measured execution times and calculated FLOPs are given.
It gives the average and bast for both.
The time gives the averaged execution time for a single FFT in case of a batched execution (an execution with more than one iteration).
They are also used to calculate the FLOPs.

The json output looks like the following.

```json

{
  "config_time": "Wed Dec 14 08:40:17 UTC 2022",
  "device": "Intel(R) FPGA Emulation Device",
  "environment": {
    "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
  },
  "errors": {
    "epsilon": 1.1920928955078125e-07,
    "residual": 0.2635231415430705
  },
  "execution_time": "Wed Dec 14 08:55:51 GMT 2022",
  "git_commit": "be1a4e9-dirty",
  "name": "FFT",
  "results": {
    "gflops_avg": {
      "unit": "GFLOP/s",
      "value": 0.2573525536079919
    },
    "gflops_min": {
      "unit": "GFLOP/s",
      "value": 0.2842073122577159
    },
    "t_avg": {
      "unit": "s",
      "value": 0.0009549545810000001
    },
    "t_min": {
      "unit": "s",
      "value": 0.00086472089
    }
  },
  "settings": {
    "Batch Size": 100,
    "Communication Type": false,
    "FFT Size": 4096,
    "Inverse": false,
    "Kernel File": false,
    "Kernel Replications": 1,
    "MPI Ranks": false,
    "Repetitions": 10,
    "Test Mode": false
  },
  "timings": {
    "execution": [
      {
        "unit": "s",
        "value": 0.151814849
      },
      {
        "unit": "s",
        "value": 0.086472089
      },
      {
        "unit": "s",
        "value": 0.089654183
      },
      {
        "unit": "s",
        "value": 0.09003793
      },
      {
        "unit": "s",
        "value": 0.089870966
      },
      {
        "unit": "s",
        "value": 0.089802216
      },
      {
        "unit": "s",
        "value": 0.089816195
      },
      {
        "unit": "s",
        "value": 0.089979618
      },
      {
        "unit": "s",
        "value": 0.090762352
      },
      {
        "unit": "s",
        "value": 0.086744183
      }
    ]
  },
  "validated": true,
  "version": "1.4"
}

```
