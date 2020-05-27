# Effective Bandwidth Benchmark for FPGA

This repository contains the effective bandwidth Benchmark for FPGA and its OpenCL kernels.
Currently only the  Intel FPGA SDK for OpenCL utility is supported.

It is a modified implementation of the
[Effective Bandwidth Benchmark](https://fs.hlrs.de/projects/par/mpi/b_eff/b_eff_3.1/#REF)
contained in the [HPC Challenge Benchmark](https://icl.utk.edu/hpcc/) suite.

A performance model for the benchmark is given in the subfolder [performance](performance).

Have a look into the [Changelog](CHANGELOG) to see recent changes in the recent version.

## Additional Dependencies

The benchmark comes next to the dependencies described in the main project with the following additional requirements for building and running:

- MPI (tested with OpenMPI 3.1.4)
- Intel OpenCL FPGA SDK 19.3  with support for external channels

## Build

The targets below can be used to build the benchmark and its kernels:

 |  Target  | Description                                    |
 | -------- | ---------------------------------------------- |
 | Network_intel     | Builds the host application                    |
 | Network_test_intel| Compile the tests and its dependencies  |
 
 More over the are additional targets to generate kernel reports and bitstreams.
 The provided kernel is optimized for the Bittware 520N board with four external
 channels.
 The code might need to be modified for other boards since the external channel descriptor
 string might be different.
 Also the number of channels is hardcoded and has to be modified for other boards.
 The kernel targets are:
 
  |  Target                        | Description                                    |
  | ------------------------------ | ---------------------------------------------- |
  | communication_bw520n_intel           | Synthesizes the kernel (takes several hours!)  |
  | communication_bw520n_report_intel    | Create an HTML report for the kernel           |
  | communication_bw520n_emulate_intel   | Create a n emulation kernel                    |
  

 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make Network_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`CHANNEL_WIDTH`  | 32          | The width of a single channel in bytes. |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

All binaries and FPGA bitstreams can be found in the `bin` directory with in the build directory.
For execution of the benchmark run:

    ./Network_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    $./Network_intel -h
    
    Implementation of the effective bandwidth benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: "1.1"
    Usage:
      ./Network_intel [OPTION...]
    
      -f, --file arg      Kernel file name
      -n, arg             Number of repetitions (default: 10)
      -l, arg             Inital looplength of Kernel (default: 32768)
          --device arg    Index of the device that has to be used. If not given
                          you will be asked which device to use if there are
                          multiple devices available. (default: -1)
          --platform arg  Index of the platform that has to be used. If not given
                          you will be asked which platform to use if there are
                          multiple platforms available. (default: -1)
      -h, --help          Print this help

    
To execute the unit and integration tests run

    ./Network_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.
The external channels are emulated through files in the `bin` folder.
They will be automatically created by the test binary.
Sending kernels will write to a file in the file system and reading kernel will
read from those files.

#### Execution with MPI

The host code also supports MPI. To run the application with two MPI processes run

    mpirun -n 2 ./fnet -f path_to_kernel.aocx
    
Platform and device selection is restricted if the application is executed with more than one rank.
In this case, the application will not ask for a platform and device if it  was not explicitly specified, but just pick one under the following criteria:
- It always picks the first available platform
- The device is picked by calculating `rank mod #devices`. This allows to run multiple processes on the same node with multiple FPGAs without conflicts in device accesses.

For example we execute the command above on a single node with two FPGAs.
Both ranks would pick the first available platform, if no platform is specified explicitly.
The used device index will be calculated with rank mod 2, so it would be 0 for the first device and 1 for the second.

Every rank will measure the kernel execution time.
The result of the measurements will be collected by rank 0 and the send bandwidth will be calculated individually for every rank.

**Note:** The MPI processes synchronize before every kernel execution using a MPI Barrier.
This might still lead to inaccuracies in the time measurements depending on the latency of the MPI network.

## Output Interpretation

The benchmark will output a result table to the standard output after execution.
This is an example output using a single rank in emulation:

            MSize      looplength            time            B/s
                1           16384     5.46779e-02     5.99292e+05
                2            8192     5.19651e-02     6.30578e+05
                4            4096     2.58565e-02     1.26730e+06
                8            2048     7.51376e-03     4.36107e+06
               16            1024     3.01288e-03     1.08760e+07
               32             512     1.66958e-03     1.96265e+07
               64             256     4.60622e-03     7.11386e+06
              128             128     1.86568e-03     1.75636e+07
              256              64     3.75094e-03     8.73594e+06
              512              32     3.81549e-03     8.58814e+06
             1024              16     3.44074e-03     9.52354e+06
             2048               8     3.83420e-03     8.54624e+06
             4096               4     3.34786e-03     9.78775e+06
            16384               2     7.84717e-03     8.35154e+06
            32768               1     7.42386e-03     8.82775e+06
            65536               1     1.40822e-02     9.30761e+06
           131072               1     1.28135e-02     2.04585e+07
           262144               1     5.52680e-02     9.48628e+06
           524288               1     9.99676e-02     1.04892e+07
          1048576               1     1.21861e-01     1.72094e+07
          2097152               1     4.20120e-01     9.98360e+06
    
    b_eff = 9.58731e+06 B/s

The table contains the measurements for all tested message sizes.
It is split into the following four columns:

- **MSize**: The size of the message in bytes
- **looplength**: The number of message exchanges that where used to measure the execution time
- **time**: Fastest execution time of the kernel with the configuration given in the first two columns over all ranks
- **GB/s**: The accumulated maximum achieved bandwidth with the current configuration using all ranks

It is possible to set the number of repetitions of the experiment. 
In this case, the best measured time will be used to calculate the bandwidth.

Under the table the calculated effective bandwidth is printed.
It is the mean of the achieved bandwidths for all used message sizes.