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
`DEFAULT_MAX_MESSAGE_SIZE`| 21 | Exclusive upper bound of the maximum message size that should be used in the benchmark. Default is 2^21=2MB|
`DEFAULT_MAX_LOOP_LENGTH`| 32768 | Maximum number of repetitions done for a single message size |
`DEFAULT_MIN_LOOP_LENGTH`| 1 | Minimum number of repetitions done for a single message size |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

## Execution

All binaries and FPGA bitstreams can be found in the `bin` directory with in the build directory.
For execution of the benchmark run:

    ./Network_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    ./Network_intel -h
    
    Implementation of the effective bandwidth benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 1.3

    MPI Version:  3.1
    Config. Time: Thu Dec 08 10:38:28 UTC 2022
    Git Commit:   86e0064-dirty

    Usage:
      ./bin/Network_intel [OPTION...]

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
      -r, arg                 Number of used kernel replications (default: 2)
          --comm-type arg     Used communication type for inter-FPGA
                              communication (default: AUTO)
          --dump-json arg     dump benchmark configuration and results to this
                              file in json format (default: )
          --test              Only test given configuration and skip execution
                              and validation
      -h, --help              Print this help
      -u, --upper arg         Maximum number of repetitions per data size
                              (default: 65536)
      -l, --lower arg         Minimum number of repetitions per data size
                              (default: 256)
          --min-size arg      Minimum Message Size (default: 0)
      -m, arg                 Maximum message size (default: 20)
      -o, arg                 Offset used before reducing repetitions (default:
                              11)
      -d, arg                 Number os steps the repetitions are decreased to
                              its minimum (default: 7)

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

               MSize             looplength               transfer                    B/s
                  64                      5            4.38310e-05            1.46015e+07
                 128                      5            7.07010e-05            1.81044e+07
                 256                      5            7.73410e-05            3.31002e+07

    b_eff = 2.19354e+07 B/s

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

The json output looks like the following.

```json

{
  "config_time": "Wed Dec 14 08:39:42 UTC 2022",
  "device": "Intel(R) FPGA Emulation Device",
  "environment": {
    "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
  },
  "errors": {},
  "execution_time": "Wed Dec 14 09:56:29 UTC 2022",
  "git_commit": "be1a4e9-dirty",
  "mpi": {
    "subversion": 1,
    "version": 3
  },
  "name": "effective bandwidth",
  "results": {
    "b_eff": {
      "unit": "B/s",
      "value": 22061624.19637537
    }
  },
  "settings": {
    "Communication Type": false,
    "Kernel File": false,
    "Kernel Replications": 2,
    "Loop Length": 5,
    "MPI Ranks": 1,
    "Message Sizes": 2,
    "Repetitions": 10,
    "Test Mode": false
  },
  "timings": {
    "6": {
      "maxCalcBW": 9880812.696844315,
      "maxMinCalculationTime": 6.4772e-05,
      "timings": [
        {
          "looplength": 5,
          "messageSize": 6,
          "timings": [
            {
              "unit": "s",
              "value": 0.010991125
            },
            {
              "unit": "s",
              "value": 8.8202e-05
            },
            {
              "unit": "s",
              "value": 0.000133323
            },
            {
              "unit": "s",
              "value": 8.5442e-05
            },
            {
              "unit": "s",
              "value": 0.000272905
            },
            {
              "unit": "s",
              "value": 0.000168784
            },
            {
              "unit": "s",
              "value": 6.4772e-05
            },
            {
              "unit": "s",
              "value": 0.000171733
            },
            {
              "unit": "s",
              "value": 0.000163393
            },
            {
              "unit": "s",
              "value": 8.0391e-05
            }
          ]
        }
      ]
    },
    "7": {
      "maxCalcBW": 19143908.348538782,
      "maxMinCalculationTime": 6.6862e-05,
      "timings": [
        {
          "looplength": 5,
          "messageSize": 7,
          "timings": [
            {
              "unit": "s",
              "value": 0.000135662
            },
            {
              "unit": "s",
              "value": 0.000119343
            },
            {
              "unit": "s",
              "value": 0.000178914
            },
            {
              "unit": "s",
              "value": 7.7691e-05
            },
            {
              "unit": "s",
              "value": 9.1922e-05
            },
            {
              "unit": "s",
              "value": 0.000259545
            },
            {
              "unit": "s",
              "value": 0.000143233
            },
            {
              "unit": "s",
              "value": 0.000149763
            },
            {
              "unit": "s",
              "value": 6.6862e-05
            },
            {
              "unit": "s",
              "value": 7.2351e-05
            }
          ]
        }
      ]
    },
    "8": {
      "maxCalcBW": 37160151.543743014,
      "maxMinCalculationTime": 6.8891e-05,
      "timings": [
        {
          "looplength": 5,
          "messageSize": 8,
          "timings": [
            {
              "unit": "s",
              "value": 0.000159723
            },
            {
              "unit": "s",
              "value": 0.000104432
            },
            {
              "unit": "s",
              "value": 0.000166953
            },
            {
              "unit": "s",
              "value": 7.7492e-05
            },
            {
              "unit": "s",
              "value": 7.8241e-05
            },
            {
              "unit": "s",
              "value": 9.5762e-05
            },
            {
              "unit": "s",
              "value": 0.000235084
            },
            {
              "unit": "s",
              "value": 0.000280265
            },
            {
              "unit": "s",
              "value": 0.000130013
            },
            {
              "unit": "s",
              "value": 6.8891e-05
            }
          ]
        }
      ]
    }
  },
  "validated": true,
  "version": "1.3"
}

```
