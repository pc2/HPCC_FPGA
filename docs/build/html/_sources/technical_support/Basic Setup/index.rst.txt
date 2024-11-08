===================
Basic Build Setup
===================

The HPCChallenge Benchmark for FPGA is fully configurable over CMake when a new build directory is created.
Although it is possible, it is highly recommended to use a different directory for the build than the source directory.
In the following, we will configure the STREAM benchmark for the use with a fictional FPGA to show the basic configuration and build process.

-------------------------------
General Benchmark Configuration
-------------------------------


The **configuration options** are implemented as CMake build parameters and can be set when creating a new CMake build directory.
We recommend to create a new build directory for a benchmark in a folder `build` in the root directory of the project.
You may want to create a folder hierarchy in there e.g. to build the STREAM benchmark create a folder `build/STREAM` and change into that new folder.
Initialize a new CMake build directory by calling

.. code-block:: bash

    cmake PATH_TO_SOURCE_DIR

where `PATH_TO_SOURCE_DIR` would be `../../STREAM` in case of STREAM (the relative path to the source directory of the target benchmark).
Some of the configuration options are the same for each benchmark and are given in the Table below. 
Especially the ``FPGA_BOARD_NAME`` is important to set, since it will specify the target board.
The ``DEFAULT_*`` options are used by the host code and can also be changed later at runtime.
The given default values will be set if no other values are given during configuration.

.. csv-table:: General Configuration Options
   :header: "Name","Default","Description"  
   :widths: 10, 10, 20                 

    ``DEFAULT_DEVICE``,-1          ,Index of the default device (-1 = ask) 
    ``DEFAULT_PLATFORM``,-1      ,Index of the default platform (-1 = ask) 
    ``DEFAULT_REPETITIONS``,10     ,Number of times the kernel will be executed 
    ``FPGA_BOARD_NAME``,p520_hpc_sg280l,Name of the target board 

Additionally, the compile options for the Intel or Xilinx compiler have to be specified. 
For the Intel compiler these are:

.. csv-table:: Intel Specific Configuration Options
   :header: "Name","Default","Description"  
   :widths: 10, 10, 20  
   
    ``AOC_FLAGS``,``-fpc -fp-relaxed -no-interleaving=default`` ,"Additional Intel AOC compiler flags that are used for kernel compilation"
    ``INTEL_CODE_GENERATION_SETTINGS``, `empty`, "Path to the settings file that will be used as input for the code generator script. It may contain additional variables or functions."

For the Xilinx compiler it is also necessary to set settings files for the compile and link step of the compiler.
The available options are given in the following table:

.. csv-table:: Xilinx Vitis Specific Configuration Options
   :header: "Name","Default","Description"  
   :widths: 10, 10, 20  

    ``XILINX_COMPILE_FLAGS`` ,``-j 40``, "Set special compiler flags like the number of used threads for compilation. "
    ``XILINX_COMPILE_SETTINGS_FILE`` , First `settings.compile.xilinx.*.ini` file found in the `settings` folder of the benchmark , "Path to the file containing compile-time settings like the target clock frequency "
    ``XILINX_LINK_SETTINGS_FILE`` , First `settings.link.xilinx.*.ini` file found in the `settings` folder of the benchmark , "Path to the file containing link settings like the mapping of the memory banks to the kernel parameters "
    ``XILINX_GENERATE_LINK_SETTINGS`` , `Yes` if the link settings file ends on `.generator.ini` `No` otherwise ," Boolean flag indicating if the link settings file will be used as a source to generate a link settings file e.g. for a given number of kernel replications"

When building a benchmark for Xilinx FPGAs double-check the path to the settings files and if they match to the target board.
The settings files follow the name convention:

    settings.[compile|link].xilinx.KERNEL_NAME.[hbm|ddr](?.generator).ini

where `KERNEL_NAME` is the name of the target OpenCL kernel file.
`hbm` or `ddr` is the type of used global memory.

All the given options can be given to CMake over the `-D` flag.

.. code-block:: bash

    cmake ../../RandomAccess -DFPGA_BOARD_NAME=my_board -D...

or after configuration using the UI with

.. code-block:: bash

    ccmake ../../RandomAccess

After configuration, there are different build targets available to proceed with the build or development.
They are shortly explained in the following.

---------------------
General Build Targets
---------------------

In general, all benchmarks offer similar build targets.
In the following they are separated into targets for the host code and for the kernels.

The host code can be build with the targets described in the table below.
`VENDOR` is either `intel` or `xilinx` depending if the Intel SDK or Xilinx Vitis should be used.
`BENCHMARK` is the host code name that is specific to the used benchmark.
You can always get an overview of the available targets by executing the following command in the build directory:

.. code-block:: bash

    make help

.. csv-table:: Host code build targets
   :header: "Target","Description"  
   :widths: 10, 30  

    BENCHMARK_VENDOR, "Builds the host application "
    BENCHMARK_test_VENDOR, "Compile the tests and its dependencies "

Moreover, there are additional targets to generate device reports and bitstreams.

The kernel targets are:
 
.. csv-table:: Device code build targets
   :header: "Target","Description"  
   :widths: 10, 30  

    BASENAME_{COMM_}VENDOR            , Synthesizes the device kernels (takes several hours!)
    BASENAME_{COMM_}report_VENDOR        , Just compile the kernels and create logs and reports
    BASENAME_{COMM_}emulate_VENDOR       , Creates the emulation kernels
  
`VENDOR` is either `intel` or `xilinx` depending if the Intel SDK or Xilinx Vitis should be used.
`BASENAME` is the name of the file containing the device code.
A benchmark can provide multiple kernel implementations and thus, these targets will be generated for every file containing kernel code.
For all benchmarks using communication between FPGAs the different communcation types are encoded into the device code file name and therefore part of target name. These are b_eff, PTRANS and LINPACK.

------------------------------------------------------
Configure and Build STREAM for a fictional Xilinx FPGA
------------------------------------------------------

We assume the code base is already checked out and we have opened a terminal in the root directory of the project.
Also, all needed dependencies are installed on the system.
To start the configuration, we first have to create a new build directory.
We want to build the STREAM benchmark, so we create a new folder `build/STREAM` and change into this directory by calling:

.. code-block:: bash

    mkdir -p build/STREAM
    cd build/STREAM

In the next step we can already configure the build.
Therefore, we call CMake with the configuration parameters we explicitly want to set.
A call could look like this:

.. code-block:: bash

    cmake ../../STREAM -DNUM_REPLICATIONS=4 \
                        -DFPGA_BOARD_NAME=fictional_fpga \
                        -DVECTOR_COUNT=16 -DGLOBAL_MEM_UNROLL=1 -DDATA_TYPE=float \
                        -DXILINX_COMPILE_SETTINGS_FILE=../../STREAM/settings/settings.compile.ini
                        -DXILINX_LINK_SETTINGS_FILE=../../STREAM/settings/settings.link.generator.ini

The number of kernel replications should be set to match the number of memory banks of the FPGA.
In this example, the FPGA has four memory banks. We want to create a kernel replication for each of them.
The name of the target FPGA board has to be determined by `xbutil scan` or a similar command.

The third row of the command defines the data type that will be used for the benchmark. 
Here we specify the data type to be `float16`, a vector data type provided by OpenCL.
This data type will contain 16 single-precision floating-point values, which again is equal to 64 bytes.
Since the memory interface of our FPGA has also a width of 64 bytes we set the unrolling to 1.
The unrolling will multiply the necessary width of the memory interface. Another possible option would have been to
use `float` by setting ``VECTOR_COUNT`` to 1 and unroll the loop 16 times by setting ``GLOBAL_MEM_UNROLL`` to 16.

The fourth row defines the location of the compile settings file that has to be used.
It usually contains the target kernel frequency and other information that might be needed by the Xilinx Vitis compiler during compilation time.

The last row defines the location to the link-time setting file.
It is used during the creation of the bitstream and contains information about the placement of the kernels in the FPGA and the mapping to the global memory.
The name of the settings file contains `*.generator:*` which is an indicator, that this settings file will be used as a template to generate a final settings file that matches
the configuration directly before synthesis.
This allows to create a single link settings file for an arbitrary number of kernel replications.

After executing the command above, the actual build files will be created.
Usually this will be Makefiles on Unix systems.
You can then start building the host code, create a report for the kernel code with the current configuration or even synthesize the kernel by using the matching build targets that are explained in the README.

---------------------------------
Using pre-defined Configurations
---------------------------------

Some benchmarks do also provide pre-defined configurations for a selection of FPGAs in the `configs` folder within their sources.
The configurations are `*.cmake` files that overwrite the default configuration options.
They can be used with CMake and the configuration option `HPCC_FPGA_CONFIG`.
As an example:

.. code-block:: bash

    cmake ../../STREAM -DHPCC_FPGA_CONFIG=../../STREAM/configs/Nallatech_520N_SP.cmake

Will use the pre-defined configuration for a Nallatech 520N board.
These configurations can also be used to document best practices in how to configure the benchmark for specific devices or architectures.
Note, that this option will overwrite all other options that may be given during CMake configuration!
To make changes on the configuration you need to unset the `HPCC_FPGA_CONFIG` variable by executing CMake as follows:

.. code-block:: bash

    cmake ../../STREAM -UHPCC_FPGA_CONFIG

After that, you can make additional changes to the build configuration.
