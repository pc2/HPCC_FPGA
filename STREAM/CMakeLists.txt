cmake_minimum_required(VERSION 3.13)
project(STREAM VERSION 2.6)

# Additional benchmark specific build parameters
set(DEFAULT_ARRAY_LENGTH 134217728 CACHE STRING "Default size of the data arrays")
set(VECTOR_COUNT 16 CACHE STRING "If greater than 1 will use OpenCL vector types in the kernels. e.g. float2")
set(GLOBAL_MEM_UNROLL 1 CACHE STRING "Unrolling factor that is used for all loops in the kernels")
set(NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")
set(DEVICE_BUFFER_SIZE 512 CACHE STRING "Buffer size in number of values that is used within the single kernel implementation.")
set(INNER_LOOP_BUFFERS ON CACHE BOOL "Put the local memory buffers inside the outer loop in the kernel code")

mark_as_advanced(INNER_LOOP_BUFFERS)

# Set the data type if not defined before to set up vector types
if (NOT DEFINED DATA_TYPE) 
    set(DATA_TYPE float)
endif()
set(HOST_DATA_TYPE cl_${DATA_TYPE})
if (VECTOR_COUNT GREATER 1)
    set(DEVICE_DATA_TYPE ${DATA_TYPE}${VECTOR_COUNT})
else()
    set(DEVICE_DATA_TYPE ${DATA_TYPE})
endif()

set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.stream_kernels_single.hbm.generator.ini)

# Use MPI if it is available
find_package(MPI)
if (MPI_FOUND)
    message(STATUS "Found MPI, so binaries will be built with MPI support")
    set(USE_MPI Yes)
endif()

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

