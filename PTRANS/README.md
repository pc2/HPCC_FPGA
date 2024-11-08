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

In addition to the dependencies defined in the main [README](../README.md), the benchmark optionally requires MKL for the CPU only implementation of the benchmark.
If MKL is not found, the benchmark can still be build without support for the CPU only execution!

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

- `transpose_diagonal`: Transposes a matrix that is distributed with the diagonal data handler
- `transpose_pq`: Transposes a matrix that is distributed with the PQ data handler. P = Q has to hold!
 
 You can build for example the host application by running
 
    mkdir build && cd build
    cmake ..
    make Transpose_intel

You will find all executables and kernel files in the `bin`
folder of your build directory.
Next to the common configuration options given in the [README](../README.md) of the benchmark suite you might want to specify the following additional options before build:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
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

    ./Transpose_xilinx -h

    Implementation of the matrix transposition benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 1.7

    MPI Version:  3.1
    Config. Time: Thu Dec 08 10:41:51 UTC 2022
    Git Commit:   86e0064-dirty

    Usage:
      ./bin/Transpose_intel [OPTION...]

      -f, --file arg            Kernel file name
      -n, arg                   Number of repetitions (default: 10)
      -i,                       Use memory Interleaving
          --skip-validation     Skip the validation of the output data. This will
                                speed up execution and helps when working with
                                special data types.
          --device arg          Index of the device that has to be used. If not
                                given you will be asked which device to use if
                                there are multiple devices available. (default: 0)
          --platform arg        Index of the platform that has to be used. If not
                                given you will be asked which platform to use if
                                there are multiple platforms available. (default:
                                0)
          --platform_str arg    Name of the platform that has to be used
                                (default: )
      -r, arg                   Number of used kernel replications (default: 2)
          --comm-type arg       Used communication type for inter-FPGA
                                communication (default: AUTO)
          --dump-json arg       dump benchmark configuration and results to this
                                file in json format (default: )
          --test                Only test given configuration and skip execution
                                and validation
      -h, --help                Print this help
      -m, arg                   Matrix size in number of blocks in one dimension
                                (default: 8)
      -b, arg                   Block size in number of values in one dimension
                                (default: 512)
      -p, arg                   Value of P that equals the width of the PQ grid
                                of FPGAs. Q is determined by the world size.
                                (default: 1)
          --distribute-buffers  Distribute buffers over memory banks. This will
                                use three memory banks instead of one for a single
                                kernel replication, but kernel replications may
                                interfere. This is an Intel only attribute, since
                                buffer placement is decided at compile time for
                                Xilinx FPGAs.
          --handler arg         Specify the used data handler that distributes
                                the data over devices and memory banks (default:
                                AUTO)

Available options for `--comm-type`:

- `CPU`: CPU only execution. MKL required.
- `IEC`: Intel external channels are used by the kernels for communication.
- `PCIE`: PCIe and MPI are used to exchange data between FPGAs over the CPU.

Possible options for `--handler`:

- `DIAG`: Diagonal distribution between FPGAs. Simplifies memory accesses by creating one-dimensional array of matrix blocks.
- `PQ`: PQ distribution of data between FPGAs. P = Q, similar to the distribution used in the LINPAK implementation.
    
To execute the unit and integration tests run

    ./Transpose_test_intel -f KERNEL_FILE_NAME
    
in the `bin` folder within the build directory.
It will run an emulation of the kernel and execute some functionality tests.
## Output Interpretation

An example output from an emulation is given below:

    Maximum error: 1.19209e-07          < 1.19209e-05
    Mach. Epsilon: 1.19209e-07

                    total time          transfer time       calc time           calc FLOPS          Memory Bandwidth    PCIe Bandwidth      
               avg: 6.05723e-02 s       1.30980e-02 s       4.74743e-02 s       3.53396e-01 GFLOP/s 4.24075e+00 GB/s    1.53708e+01 GB/s    
              best: 4.69977e-02 s       1.05343e-02 s       3.64633e-02 s       4.60112e-01 GFLOP/s 5.52134e+00 GB/s    1.91115e+01 GB/s    

The output gives the average and best calculation time for the transposition and important derived metrics based on these times.
For the average and best timings, we have the following columns:

- `total [s]`: Total execution time in seconds of a whole repetition of the experiment that includes transfer and calcuation time.
- `transfer [s]`: Time in seconds that is required to transfer the data buffers to a memory location that can be accessed by the kernels on the FPGA board.
- `calc [s]`: Time in seconds to execute a single repetition of the matrix transposition also including communication between devices.
- `calc FLOPS`: Derived floating-point operations per second based on the calculation time.
- `Mem [B/s]`: Derived bandwidth of the memory that is accessed by the FPGA kernels during calculation based on the calculation time.
- `PCIe [B/s]`: Derived bandwidth of the transfer interface that is used to copy the buffers to a memory location accessible by the FPGA based on the transfer time.

The `Maximum Error` field shows the largest error that was computed.
Since the arithmetic intensity of the algorithm is quite low and only one addition is required to calculate one value of the result matrix, the error should be close to the machine epsilon, which depends on the chosen data type.
The second number in the row is the validation threshold which is depending on the machine epsilon.
The validation is successful, if the maximum error is below this threshold.
The machine epsilon is given in the row below with `Mach. Epsilon`.
Moreover, the total time that was needed for the validation of the result is given, which is just a debug information.
The very last column summarizes the result: The last row will show `Validation: SUCCESS!` if the validation succeeded and the error is below the tolerated threshold.

The json output looks like the following.

```json

{
  "config_time": "Wed Dec 14 08:42:29 UTC 2022",
  "device": "Intel(R) FPGA Emulation Device",
  "environment": {
    "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
  },
  "errors": {
    "epsilon": 1.1920928955078125e-07,
    "max_error": 199.96849060058594
  },
  "execution_time": "Wed Dec 14 09:57:30 UTC 2022",
  "git_commit": "be1a4e9-dirty",
  "mpi": {
    "subversion": 1,
    "version": 3
  },
  "name": "matrix transposition",
  "results": {
    "avg_calc_flops": {
      "unit": "GFLOP/s",
      "value": 0.011002914958427963
    },
    "avg_calc_t": {
      "unit": "s",
      "value": 1.524797389
    },
    "avg_mem_bandwidth": {
      "unit": "GB/s",
      "value": 0.13203497950113555
    },
    "avg_t": {
      "unit": "s",
      "value": 1.5332141689999998
    },
    "avg_transfer_bandwidth": {
      "unit": "GB/s",
      "value": 23.919669042080226
    },
    "avg_transfer_t": {
      "unit": "s",
      "value": 0.00841678
    },
    "max_calc_flops": {
      "unit": "GFLOP/s",
      "value": 0.011002914958427963
    },
    "max_mem_bandwidth": {
      "unit": "GB/s",
      "value": 0.13203497950113555
    },
    "max_transfer_bandwidth": {
      "unit": "GB/s",
      "value": 23.919669042080226
    },
    "min_calc_t": {
      "unit": "s",
      "value": 1.524797389
    },
    "min_t": {
      "unit": "s",
      "value": 1.5332141689999998
    },
    "min_transfer_t": {
      "unit": "s",
      "value": 0.00841678
    }
  },
  "settings": {
    "Block Size": 512,
    "Communication Type": false,
    "Data Handler": false,
    "Dist. Buffers": false,
    "FPGA Torus": {
      "P": 1,
      "Q": 3
    },
    "Kernel File": false,
    "Kernel Replications": 2,
    "MPI Ranks": 3,
    "Matrix Size": 4096,
    "Repetitions": 1,
    "Test Mode": false
  },
  "timings": {
    "calculation": [
      {
        "unit": "s",
        "value": 1.523696949
      }
    ],
    "transfer": [
      {
        "unit": "s",
        "value": 0.008189295
      }
    ]
  },
  "validated": false,
  "version": "1.7"
}

```
