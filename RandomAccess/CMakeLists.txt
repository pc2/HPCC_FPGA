cmake_minimum_required(VERSION 3.13)
project(RandomAccess VERSION 2.2)

# Additional benchmark specific build parameters
set(DEFAULT_ARRAY_LENGTH 536870912 CACHE STRING "Default size of the data arrays")
set(PARALLEL_MEM_ACCESSES 1 CACHE STRING "Unrolling factor that is used for all loops in the kernels that access the global memory.")
set(NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")
set(DEVICE_BUFFER_SIZE 1 CACHE STRING "Buffer size in number of values that is used within the single kernel implementation.")
set(COMBINE_LOOPS Yes CACHE BOOL "If enabled this will combine the address calculation loop and the load darta loop to a single loop. This can improve the performance when all loops are running sequentially")

set(DATA_TYPE long)
set(HOST_DATA_TYPE cl_ulong)
set(HOST_DATA_TYPE_SIGNED cl_long)
set(DEVICE_DATA_TYPE long)
set(DEVICE_DATA_TYPE_UNSIGNED ulong)

if (NUM_REPLICATIONS EQUAL 1)
    set(SINGLE_KERNEL Yes)
endif()

include(${CMAKE_SOURCE_DIR}/../cmake/general_benchmark_build_setup.cmake)

unset(DATA_TYPE CACHE)
