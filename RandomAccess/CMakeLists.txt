cmake_minimum_required(VERSION 3.1)
project(RandomAccess VERSION 2.0.2.1
        DESCRIPTION "OpenCL RandomAccess Benchmark for FPGA")

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../extern/hlslib/cmake)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

find_package(OpenMP)
find_package(IntelFPGAOpenCL)
find_package(Vitis)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

set(DEFAULT_REPETITIONS 10 CACHE STRING "Default number of repetitions")
set(DEFAULT_ARRAY_LENGTH 536870912 CACHE STRING "Default size of the data arrays")
set(DEFAULT_DEVICE -1 CACHE STRING "Index of the default device to use")
set(DEFAULT_PLATFORM -1 CACHE STRING "Index of the default platform to use")
set(PARALLEL_MEM_ACCESSES 1 CACHE STRING "Unrolling factor that is used for all loops in the kernels that access the global memory.")
set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Name of the target FPGA board")
set(NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")
set(DEVICE_BUFFER_SIZE 1 CACHE STRING "Buffer size in number of values that is used within the single kernel implementation.")
set(COMBINE_LOOPS Yes CACHE BOOL "If enabled this will combine the address calculation loop and the load darta loop to a single loop. This can improve the performance when all loops are running sequentially")
set(USE_SVM No CACHE BOOL "Use coarse grained SVM instead of loading the buffer on the FPGA before execution. Device needs to support this feature.")

if (VITIS_FOUND)
    set(XILINX_GENERATE_LINK_SETTINGS Yes CACHE BOOL "Generate the link settings depending on the number of replications for the kernels")
    set(XILINX_LINK_SETTINGS_FILE "${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.random_access_single.generator.ini" CACHE STRING "The link settings file that should be used when generating is disabled")
    set(XILINX_COMPILE_FLAGS "-j 40" CACHE STRING "Additional compile flags for the v++ compiler")
    set(XILINX_COMPILE_SETTINGS "${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.ini" CACHE STRING "Compile settings file for the v++ compiler")
    separate_arguments(XILINX_COMPILE_FLAGS)
endif()

if (INTELFPGAOPENCL_FOUND)
    set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "Used flags for the AOC compiler")
    separate_arguments(AOC_FLAGS)
endif()

set(HOST_DATA_TYPE cl_ulong)
set(HOST_DATA_TYPE_SIGNED cl_long)
set(DEVICE_DATA_TYPE long)
set(DEVICE_DATA_TYPE_UNSIGNED ulong)

if (NUM_REPLICATIONS EQUAL 1)
    set(SINGLE_KERNEL Yes)
endif()

configure_file(
        "${CMAKE_SOURCE_DIR}/src/common/parameters.h.in"
        "${CMAKE_BINARY_DIR}/src/common/parameters.h"
)
include_directories(${CMAKE_BINARY_DIR}/src/common)

add_subdirectory(src/device)
add_subdirectory(src/host)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
