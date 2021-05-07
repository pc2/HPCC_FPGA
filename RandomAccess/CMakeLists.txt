cmake_minimum_required(VERSION 3.13)
project(RandomAccess VERSION 2.4)

# Additional benchmark specific build parameters
set(DEFAULT_ARRAY_LENGTH_LOG 29 CACHE STRING "Default size of the data arrays")
set(HPCC_FPGA_RA_NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")
set(HPCC_FPGA_RA_DEVICE_BUFFER_SIZE_LOG 10 CACHE STRING "Buffer size in number of values that is used within the single kernel implementation.")
set(HPCC_FPGA_RA_INTEL_USE_PRAGMA_IVDEP No CACHE BOOL "Use the pragma ivdep in the main loop to remove dependency of global memory reads and writes. This might lead to non validating kernel.") 
set(HPCC_FPGA_RA_RNG_COUNT_LOG 7 CACHE BOOL "Log2 of the number of random number generators that will be used concurrently")
set(HPCC_FPGA_RA_RNG_DISTANCE 5 CACHE BOOL "Distance between RNGs in shift register. Used to relax data dependencies and increase clock frequency")
set(HPCC_FPGA_RA_GLOBAL_MEM_UNROLL_LOG 3 CACHE BOOL "Log2 of the global memory burst size in number of values that can be read from memory in a single clock cycle")

set(DATA_TYPE long)
set(HOST_DATA_TYPE cl_ulong)
set(HOST_DATA_TYPE_SIGNED cl_long)
set(DEVICE_DATA_TYPE long)
set(DEVICE_DATA_TYPE_UNSIGNED ulong)

if (NUM_REPLICATIONS EQUAL 1)
    set(SINGLE_KERNEL Yes)
endif()

# Use MPI if it is available
find_package(MPI)
if (MPI_FOUND)
    message(STATUS "Found MPI, so binaries will be built with MPI support")
    set(USE_MPI Yes)
endif()

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

unset(DATA_TYPE CACHE)
