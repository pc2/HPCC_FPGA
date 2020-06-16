cmake_minimum_required(VERSION 3.13)
project(FFT VERSION 1.1)

set(DEFAULT_ITERATIONS 100 CACHE STRING "Default number of iterations that is done with a single kernel execution")
set(HOST_DATA_TYPE cl_float CACHE STRING "Data type used by the host code. Should match the data type of the used FFT")

set(FFT_KERNEL_NAME fft1d CACHE STRING "Name of the kernel that is used for calculation")
set(FETCH_KERNEL_NAME fetch CACHE STRING "Name of the kernel that is used to fetch data from global memory")
set(LOG_FFT_SIZE 12 CACHE STRING "Log2 of the used FFT size")
set(FFT_UNROLL 8 CACHE STRING "Amount of global memory unrolling of the kernel. Will be used by the host to calculate NDRange sizes")

set(DATA_TYPE float)
include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

unset(DATA_TYPE CACHE)

