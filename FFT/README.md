# FFT Benchmark for FPGA

This repository contains the FFT Benchmark for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

It is based on the FFT benchmark of the [HPC Challenge Benchmark](https://icl.utk.edu/hpcc/) suite.
The FFT1D reference implementation is used for the kernel code.

## Additional Dependencies

The benchmark needs no additional dependencies than the ones given in the main [README](../README.md).

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels:

 |  Target  | Description                                    |
 | -------- | ---------------------------------------------- |
 | fFFT     | Builds the host application                    |
 | Google_Tests_run| Compile the tests and its dependencies  |
 
 More over the are additional targets to generate kernel reports and bitstreams.
 The provided kernel is optimized for Stratix 10 with 512bit LSUs.
 The kernel targets are:
 
  |  Target  | Description                                    |
  | -------- | ---------------------------------------------- |
  | fft1d_float_8_intel          | Synthesizes the kernel (takes several hours!)  |
  | fft1d_float_8_report_intel   | Create an HTML report for the kernel    |
  | fft1d_float_8_emulate_intel  | Create a n emulation kernel             |
  
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make fFFT

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`DEFAULT_ITERATIONS`| 100          | Default number of iterations that is done with a single kernel execution|
`LOG_FFT_SIZE`   | 12          | Log2 of the FFT Size that has to be used i.e. 3 leads to a FFT Size of 2^3=8|

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

For execution of the benchmark run:

    ./fFFT -f path_to_kernel.aocx
    
For more information on available input parameters run

    $./fFFT -h
    
    Implementation of the FFT benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: "1.0"
    Usage:
      ./fFFT [OPTION...]
    
      -f, --file arg      Kernel file name
      -n, arg             Number of repetitions (default: 10)
      -i, arg             Multiplier for the used data size that will be i *
                          FFT_SIZE (default: 100)
          --inverse       If set, the inverse FFT is calculated instead
          --device arg    Index of the device that has to be used. If not given
                          you will be asked which device to use if there are
                          multiple devices available. (default: -1)
          --platform arg  Index of the platform that has to be used. If not given
                          you will be asked which platform to use if there are
                          multiple platforms available. (default: -1)
      -h, --help          Print this help
    
To execute the unit and integration tests run

    ./Google_Tests_run
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Output Interpretation

The benchmark will print the following two tables to standard output after execution:

       res. error    mach. eps
      2.67000e-01  1.19209e-07
    
                           avg         best
       Time in s:  7.56801e-03  7.07241e-03
          GFLOPS:  3.24735e-02  3.47491e-02
          
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
