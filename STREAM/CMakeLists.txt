cmake_minimum_required(VERSION 3.1)
project(STREAM VERSION 2.1.4.1)

# Additional benchmark specific build parameters
set(DEFAULT_ARRAY_LENGTH 134217728 CACHE STRING "Default size of the data arrays")
set(VECTOR_COUNT 16 CACHE STRING "If greater than 1 will use OpenCL vector types in the kernels. e.g. float2")
set(GLOBAL_MEM_UNROLL 1 CACHE STRING "Unrolling factor that is used for all loops in the kernels")
set(NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")
set(DEVICE_BUFFER_SIZE 512 CACHE STRING "Buffer size in number of values that is used within the single kernel implementation.")
set(INNER_LOOP_BUFFERS ON CACHE BOOL "Put the local memory buffers inside the outer loop in the kernel code")
set(USE_SVM No CACHE BOOL "Use SVM pointers instead of creating buffers on the board and transferring the data there before execution.")

# Set the data type since optional vector types are used
set(DATA_TYPE float)
set(HOST_DATA_TYPE cl_${DATA_TYPE})
if (VECTOR_COUNT GREATER 1)
    set(DEVICE_DATA_TYPE ${DATA_TYPE}${VECTOR_COUNT})
else()
    set(DEVICE_DATA_TYPE ${DATA_TYPE})
endif()

set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.stream_kernels_single.hbm.generator.ini)

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

