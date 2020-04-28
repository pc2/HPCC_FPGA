cmake_minimum_required(VERSION 2.8.12)
project(FFT VERSION 1.0)

set(DEFAULT_REPETITIONS 10 CACHE STRING "Default number of repetitions")
set(DEFAULT_ITERATIONS 100 CACHE STRING "Default number of iterations that is done with a single kernel execution")
set(HOST_DATA_TYPE cl_float CACHE STRING "Data type used by the host code. Should match the data type of the used FFT")
set(DEFAULT_DEVICE -1 CACHE STRING "Index of the default device to use")
set(DEFAULT_PLATFORM -1 CACHE STRING "Index of the default platform to use")
set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Name of the target FPGA board")

set(AOC_FLAGS "-fpc -fp-relaxed" CACHE STRING "Used flags for the AOC compiler")
separate_arguments(AOC_FLAGS)


set(FFT_KERNEL_NAME fft1d CACHE STRING "Name of the kernel that is used for calculation")
set(FETCH_KERNEL_NAME fetch CACHE STRING "Name of the kernel that is used to fetch data from global memory")
set(LOG_FFT_SIZE 12 CACHE STRING "Log2 of the used FFT size")
set(FFT_UNROLL 8 CACHE STRING "Amount of global memory unrolling of the kernel. Will be used by the host to calculate NDRange sizes")

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

