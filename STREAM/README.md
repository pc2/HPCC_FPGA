# STREAM for FPGA

This repository contains the STREAM benchmark for FPGA and its OpenCL kernels.
This version adds basic support for the Xilinx Vitis toolcchain.

The implementation is based on the STREAM benchmark 5.10 by John D. McCaplin, Ph.D.
available at [https://www.cs.virginia.edu/stream/](https://www.cs.virginia.edu/stream/).

## Build

CMake is used as the build system.
The targets below can be used to build the benchmark and its kernels:

 |  Target               | Description                                    |
 | --------------------- | ---------------------------------------------- |
 | STREAM_FPGA_`VENDOR`     | Builds the host application linking with the Intel SDK|
 | STREAM_FPGA_test_`VENDOR`            | Compile the tests and its dependencies linking with the Intel SDK  |
 
 More over there are additional targets to generate kernel reports and bitstreams.
 The provided kernel is optimized for the Bittware 520N board with four external
 channels.
 The code might need to be modified for other boards since the external channel descriptor
 string might be different.
 Also the number of channels is hardcoded and has to be modified for other boards.
 The kernel targets are:
 
  |  Target                        | Description                                    |
  | ------------------------------ | ---------------------------------------------- |
  | stream_kernels_`VENDOR`                | Synthesizes the kernel (takes several hours!)  |
  | stream_kernels_report_`VENDOR`         | Just compile kernel and create logs and reports |
  | stream_kernels_emulate_`VENDOR`          | Create a n emulation kernel                    |
  | stream_kernels_single_`VENDOR`                | Synthesizes the kernel (takes several hours!)  |
  | stream_kernels_single_report_`VENDOR`          | Just compile kernel and create logs and reports |
  | stream_kernels_single_emulate_`VENDOR`          | Create a n emulation kernel                    |
  
For the host code as well as the kernels `VENDOR` can be `intel` or `xilinx`.
The report target for xilinx is missing but reports will be generated when the kernel is synthesized.
The `stream_kernels_single` targets build a single kernel for all four vector operations.
In that case the operation will be defined by an additional operation type input parameter.
To use this kernel, run the host code with the parameter `--single-kernel`.

 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make STREAM_FPGA_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`DATA_TYPE`      | float       | Data type used for host and device code |
`VECTOR_COUNT`   | 16           | If >1 OpenCL vector types of the given size are used in the device code |
`DEFAULT_ARRAY_LENGTH`| 134217728 | Length of each input array |
`GLOBAL_MEM_UNROLL`| 1        | Loop unrolling factor for all loops in the device code |
`NUM_REPLICATIONS`| 1        | Replicates the kernels the given number of times |
`DEVICE_BUFFER_SIZE`| 16384        | Number of values that are stored in the local memory in the single kernel approach |

Moreover the environment variable `INTELFPGAOCLSDKROOT` has to be set to the root
of the Intel FPGA SDK installation.

Additionally it is possible to set the used compiler and other build tools 
in the `CMakeCache.txt` located in the build directory after running cmake.

## Execution

For execution of the benchmark run:

    ./STREAM_FPGA_intel -f path_to_kernel.aocx
    
For more information on available input parameters run

    $./STREAM_FPGA_intel -h

    Implementation of the STREAM benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 2.6

    MPI Version:  3.1
    Config. Time: Thu Dec 08 10:43:26 UTC 2022
    Git Commit:   86e0064-dirty

    Usage:
      ./bin/STREAM_FPGA_intel [OPTION...]

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
      -s, arg                 Size of the data arrays (default: 134217728)
          --multi-kernel      Use the legacy multi kernel implementation

To execute the unit and integration tests for Intel devices run

    CL_CONTEXT_EMULATOR_DEVICE=1 ./STREAM_FPGA_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.

## Result interpretation

The output of the host application is similar to the original STREAM benchmark:

    Function            Best Rate           Avg time            Min time            Max time
    PCI_write           2.68152e+04 MB/s    6.36535e-02 s       6.00633e-02 s       8.45139e-02 s
    PCI_read            2.47220e+04 MB/s    6.72553e-02 s       6.51490e-02 s       6.82519e-02 s
    Copy                4.75583e+04 MB/s    2.32275e-02 s       2.25774e-02 s       2.55071e-02 s
    Scale               5.35745e+04 MB/s    2.13423e-02 s       2.00420e-02 s       2.42722e-02 s
    Add                 5.36221e+04 MB/s    3.33479e-02 s       3.00364e-02 s       3.68116e-02 s
    Triad               4.84564e+04 MB/s    3.46477e-02 s       3.32384e-02 s       3.70085e-02 s

In addition it also measures the bandwidth of the connection between host and
device. It is distinguished between writing to and reading from the devices
memory.
The buffers are written to the device before every iteration and read back
after each iteration.

## Exemplary Results

The benchmark was executed on Bittware 520N cards for different Intel® Quartus® Prime versions.
The detailed results of the runs are given in [results.txt](results.txt) and as
CSV files in the subfolder `csv_result_export`.

#### Single Precision

![Single precision results](csv_result_export/sp_plot.jpeg)

#### Double Precision

![Double precision results](csv_result_export/dp_plot.jpeg)

#### Usage of the Global Ring

It is possible to force a ring interconnect for the global memory with the compiler command
`-global-ring`. To test the impact of this type of interconnect, the benchmark was also 
synthesized with the additional parameters `-global-ring -duplicate-ring` for all SDK versions
supporting this option.

The raw data of these runs can be found in the folder `csv_result_export`.

##### Single Precision
![Single precision results](csv_result_export/sp_global_ring_plot.jpeg)

##### Double Precision
![Double precision results](csv_result_export/dp_global_ring_plot.jpeg)

```json

{
  "config_time": "Wed Dec 14 08:43:42 UTC 2022",
  "device": "Intel(R) FPGA Emulation Device",
  "environment": {
    "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
  },
  "errors": {
    "a_average_error": 0,
    "a_average_relative_error": 0,
    "a_expected": 1153300692992,
    "b_average_error": 0,
    "b_average_relative_error": 0,
    "b_expected": 230660145152,
    "c_average_error": 0,
    "c_average_relative_error": 0,
    "c_expected": 307546849280,
    "epsilon": 1.1920928955078125e-07
  },
  "execution_time": "Wed Dec 14 09:29:17 UTC 2022",
  "git_commit": "be1a4e9-dirty",
  "mpi": {
    "subversion": 1,
    "version": 3
  },
  "name": "STREAM",
  "results": {
    "Add_avg_t": {
      "unit": "s",
      "value": 0.0530118015
    },
    "Add_best_rate": {
      "unit": "MB/s",
      "value": 30506.44534004568
    },
    "Add_max_t": {
      "unit": "s",
      "value": 0.053374228
    },
    "Add_min_t": {
      "unit": "s",
      "value": 0.052795818
    },
    "Copy_avg_t": {
      "unit": "s",
      "value": 0.0389517071
    },
    "Copy_best_rate": {
      "unit": "MB/s",
      "value": 27731.67753145461
    },
    "Copy_max_t": {
      "unit": "s",
      "value": 0.040187928
    },
    "Copy_min_t": {
      "unit": "s",
      "value": 0.038718964
    },
    "PCI_read_avg_t": {
      "unit": "s",
      "value": 0.0597715322
    },
    "PCI_read_best_rate": {
      "unit": "MB/s",
      "value": 27479.82304062059
    },
    "PCI_read_max_t": {
      "unit": "s",
      "value": 0.063351971
    },
    "PCI_read_min_t": {
      "unit": "s",
      "value": 0.058610739
    },
    "PCI_write_avg_t": {
      "unit": "s",
      "value": 0.0685080558
    },
    "PCI_write_best_rate": {
      "unit": "MB/s",
      "value": 25765.843668891466
    },
    "PCI_write_max_t": {
      "unit": "s",
      "value": 0.120777629
    },
    "PCI_write_min_t": {
      "unit": "s",
      "value": 0.062509606
    },
    "Scale_avg_t": {
      "unit": "s",
      "value": 0.03978323250000001
    },
    "Scale_best_rate": {
      "unit": "MB/s",
      "value": 27084.469403573872
    },
    "Scale_max_t": {
      "unit": "s",
      "value": 0.039983335
    },
    "Scale_min_t": {
      "unit": "s",
      "value": 0.039644189
    },
    "Triad_avg_t": {
      "unit": "s",
      "value": 0.052600337100000005
    },
    "Triad_best_rate": {
      "unit": "MB/s",
      "value": 30701.997665172144
    },
    "Triad_max_t": {
      "unit": "s",
      "value": 0.052735936
    },
    "Triad_min_t": {
      "unit": "s",
      "value": 0.052459542
    }
  },
  "settings": {
    "Array Size": 134217728,
    "Communication Type": false,
    "Data Type": false,
    "Kernel File": false,
    "Kernel Replications": 4,
    "Kernel Type": false,
    "MPI Ranks": 1,
    "Repetitions": 10,
    "Test Mode": false
  },
  "timings": {
    "Add": [
      {
        "unit": "s",
        "value": 0.052848008
      },
      {
        "unit": "s",
        "value": 0.052795818
      },
      {
        "unit": "s",
        "value": 0.053294617
      },
      {
        "unit": "s",
        "value": 0.053374228
      },
      {
        "unit": "s",
        "value": 0.052812528
      },
      {
        "unit": "s",
        "value": 0.053091652
      },
      {
        "unit": "s",
        "value": 0.052962381
      },
      {
        "unit": "s",
        "value": 0.052992892
      },
      {
        "unit": "s",
        "value": 0.052880469
      },
      {
        "unit": "s",
        "value": 0.053065422
      }
    ],
    "Copy": [
      {
        "unit": "s",
        "value": 0.040187928
      },
      {
        "unit": "s",
        "value": 0.038718964
      },
      {
        "unit": "s",
        "value": 0.038728084
      },
      {
        "unit": "s",
        "value": 0.038760534
      },
      {
        "unit": "s",
        "value": 0.038793734
      },
      {
        "unit": "s",
        "value": 0.039005018
      },
      {
        "unit": "s",
        "value": 0.038862845
      },
      {
        "unit": "s",
        "value": 0.038731043
      },
      {
        "unit": "s",
        "value": 0.038891176
      },
      {
        "unit": "s",
        "value": 0.038837745
      }
    ],
    "PCI_read": [
      {
        "unit": "s",
        "value": 0.058610739
      },
      {
        "unit": "s",
        "value": 0.059211539
      },
      {
        "unit": "s",
        "value": 0.059094178
      },
      {
        "unit": "s",
        "value": 0.063351971
      },
      {
        "unit": "s",
        "value": 0.059738369
      },
      {
        "unit": "s",
        "value": 0.059645487
      },
      {
        "unit": "s",
        "value": 0.059697218
      },
      {
        "unit": "s",
        "value": 0.059381852
      },
      {
        "unit": "s",
        "value": 0.059468254
      },
      {
        "unit": "s",
        "value": 0.059515715
      }
    ],
    "PCI_write": [
      {
        "unit": "s",
        "value": 0.120777629
      },
      {
        "unit": "s",
        "value": 0.062600188
      },
      {
        "unit": "s",
        "value": 0.062606179
      },
      {
        "unit": "s",
        "value": 0.062711891
      },
      {
        "unit": "s",
        "value": 0.062509606
      },
      {
        "unit": "s",
        "value": 0.062803592
      },
      {
        "unit": "s",
        "value": 0.062787151
      },
      {
        "unit": "s",
        "value": 0.062679419
      },
      {
        "unit": "s",
        "value": 0.06271488
      },
      {
        "unit": "s",
        "value": 0.062890023
      }
    ],
    "Scale": [
      {
        "unit": "s",
        "value": 0.039983335
      },
      {
        "unit": "s",
        "value": 0.039644189
      },
      {
        "unit": "s",
        "value": 0.039831532
      },
      {
        "unit": "s",
        "value": 0.039766591
      },
      {
        "unit": "s",
        "value": 0.039660679
      },
      {
        "unit": "s",
        "value": 0.039933614
      },
      {
        "unit": "s",
        "value": 0.039789862
      },
      {
        "unit": "s",
        "value": 0.03967413
      },
      {
        "unit": "s",
        "value": 0.039722601
      },
      {
        "unit": "s",
        "value": 0.039825792
      }
    ],
    "Triad": [
      {
        "unit": "s",
        "value": 0.052583184
      },
      {
        "unit": "s",
        "value": 0.052564403
      },
      {
        "unit": "s",
        "value": 0.052735936
      },
      {
        "unit": "s",
        "value": 0.052644865
      },
      {
        "unit": "s",
        "value": 0.052699956
      },
      {
        "unit": "s",
        "value": 0.052459542
      },
      {
        "unit": "s",
        "value": 0.052657585
      },
      {
        "unit": "s",
        "value": 0.052493212
      },
      {
        "unit": "s",
        "value": 0.052600984
      },
      {
        "unit": "s",
        "value": 0.052563704
      }
    ]
  },
  "validated": true,
  "version": "2.6"
}

```
