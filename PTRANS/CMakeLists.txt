cmake_minimum_required(VERSION 2.8.4)
project(fTrans)

set(VERSION 1.0.0)
set(KERNEL_NAME transpose CACHE STRING "Name of the OpenCL kernel")
set(DEFAULT_REPETITIONS 10 CACHE STRING "Default number of repetitions")
set(DEFAULT_MATRIX_SIZE 4096 CACHE STRING "Default size of the used matrices")
set(DEFAULT_PLATFORM -1 CACHE STRING "Index of the default platform to use")
set(DEFAULT_DEVICE -1 CACHE STRING "Index of the default device to use")
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel")
set(GLOBAL_MEM_UNROLL 16 CACHE STRING "Unrolling factor used to stream data")
set(DATA_TYPE float CACHE STRING "Data type used for calculation")
set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Name of the target FPGA board")

set(AOC_FLAGS "-fpc -fp-relaxed" CACHE STRING "Used flags for the AOC compiler")
separate_arguments(AOC_FLAGS)

set(HOST_DATA_TYPE cl_${DATA_TYPE})
set(DEVICE_DATA_TYPE ${DATA_TYPE})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../extern/hlslib/cmake)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

configure_file(
        "${CMAKE_SOURCE_DIR}/src/common/parameters.h.in"
        "${CMAKE_BINARY_DIR}/src/common/parameters.h"
)

include_directories(${CMAKE_BINARY_DIR}/src/common)

find_package(IntelFPGAOpenCL REQUIRED)

add_subdirectory(src/device)
add_subdirectory(src/host)
add_subdirectory(tests)

