=================================
Structure of the Benchmark Suite
=================================

The structure of the sources of the benchmark suite aims to reduce the redundancy of code and documentation.
Therefore, we describe the basic structure in this section to make it easier to get started.

--------------
Root Directory
--------------

In general, the root folder of the project contains separate folders for every benchmark. 
Additionally, there are folders for commonly used code and scripts that will be explained in the following.
A brief description of the benchmark suite is also given in the `README.md <https://github.com/pc2/HPCC_FPGA#readme>`_ with further help on how to get started and on vendor compatability.

**b_eff, FFT, GEMM, LINPACK, PTRANS, RandomAccess, STREAM**: 
    These are the folders for each of the seven benchmarks of the suite.
    The contents of these folders are organized very similar. The contents of these *benchmark folders* will be explained in more detail in another section.
    One main design principle of the benchmark suite is, that only a single benchmark can be configured at a time, because the configurations for two benchmarks may be incompatible.
    However, it is easly possible to configure and build multiple benchmarks from the same sources by creating separate build directories for each benchmark. 

**cmake**: 
    Contains CMake scripts that are used by all benchmarks to generate build targets for the kernel codes and testing. Files in here may require changes, if 
    the kernel build process needs to be changed or additional parameters are required to build the kernels of a benchmark.

**docs**: 
    This folder contains the Sphinx documentation and a Doxyfile to generate the code documentation. You may hae to enable the HTML output in the Doxyfile to
    create the classic HTML documentation.

**extern**: 
    This folder contains references to external dependencies. These are projects that are used within the benchmarks, i.e. for input parameter parsing. The
    dependencies are managed using CMakes fetch support and will be checked out during the first configuration of a benchmark, so internet connection will be required.

**scripts**: 
    Contains different helper scripts for code generation, that are used within the CMake build process of ech benchmark. Moreover it contains useful scripts 
    to parse the benchmark outputs to CSV files or simply power measurements and testing.

**shared**: 
    This folder contains the host source code for the host that is commonly used by all benchmarks. It implements the basic fucntionality of all benchmarks by going through
    the steps *parameter parsing*, *Setup and programming of FPGAs*, *input data generation*, *execution* and *validation*. The individual behavior in these five stages is defined in the benchmark specific code in the *benchmark folders*.


-----------------
Benchmark Folders
-----------------

The folders for each benchmark are structured in the same way and all contain the same sub-folders.
A brief description of the benchmark and its configuration parameters is given in the README.md file.
Every benchmark folder also contains a CHANGELOG file which documents code changes and changes in the benchmark rules.

**configs**: 
    This folder contains CMake files that contain predefined configurations for selected FPGA boards. This simplifies the coniguration for these boards    
    because the different parameters do not need to be modified manually. Instead, they can be used with the ``HPCC_FPGA_CONFIG`` flag to configure a build for one of the boards.

**scripts**: 
    Contains helper scripts to build or execute the benchmark on an HPC system with WLM. The scripts can be used as a base to create own scripts for the target machine. Most likely they will not work out of the box since they are written for a specific system.

**settings**: 
    For some benchmarks it is required to give additional information to the build system. This can be done with settings files in this folder. For example it contains the compile and link settings for specific Xilinx devices. Settings files are device specific, so they may need to be modified.

**src**: 
    Contains the source code for the host executable - which makes heavily used of the shared code in the *shared* folder in the root directory. In most cases, the host code consists of a *_benchmark.cpp* file which contains the benchmark specific implementations of the data generation, validation and parameter parsing. 
    The *execution.cpp* contains a single *calculate* function which executes the benchmark kernels and measures the time.

**tests**: 
    Every benchmark comes with Unit tests targeting basic functionality of the host code and with a focus on the correctness of the device (OpenCL) code.
    
