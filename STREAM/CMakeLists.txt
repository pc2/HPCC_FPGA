cmake_minimum_required(VERSION 3.1)
project(STREAM VERSION 2.1.4.1
        DESCRIPTION "OpenCL STREAM Benchmark for FPGA")

set (CMAKE_CXX_STANDARD 11)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

set(DEFAULT_REPETITIONS 10 CACHE STRING "Default number of repetitions")
set(DEFAULT_ARRAY_LENGTH 134217728 CACHE STRING "Default size of the data arrays")
set(DATA_TYPE float CACHE STRING "Data type used for host and device")
set(VECTOR_COUNT 16 CACHE STRING "If greater than 1 will use OpenCL vector types in the kernels. e.g. float2")
set(DEFAULT_DEVICE -1 CACHE STRING "Index of the default device to use")
set(DEFAULT_PLATFORM -1 CACHE STRING "Index of the default platform to use")
set(GLOBAL_MEM_UNROLL 1 CACHE STRING "Unrolling factor that is used for all loops in the kernels")
set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Name of the target FPGA board")
set(NUM_REPLICATIONS 4 CACHE STRING "Number of times the kernels will be replicated")
set(DEVICE_BUFFER_SIZE 512 CACHE STRING "Buffer size in number of values that is used within the single kernel implementation.")
set(INNER_LOOP_BUFFERS ON CACHE BOOL "Put the local memory buffers inside the outer loop in the kernel code")
set(USE_SVM No CACHE BOOL "Use SVM pointers instead of creating buffers on the board and transferring the data there before execution.")
set(XILINX_GENERATE_LINK_SETTINGS Yes CACHE BOOL "Generate the link settings depending on the number of replications for the kernels")
set(XILINX_LINK_SETTINGS_FILE "${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.stream_kernels.ddr.ini" CACHE STRING "The link settings file that should be used when generating is disabled")
set(XILINX_COMPILE_FLAGS "-j 40" CACHE STRING "Additional compile flags for the v++ compiler")
set(XILINX_COMPILE_SETTINGS "${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.ini" CACHE STRING "Compile settings file for the v++ compiler")
separate_arguments(XILINX_COMPILE_FLAGS)

set(HOST_DATA_TYPE cl_${DATA_TYPE})
if (VECTOR_COUNT GREATER 1)
    set(DEVICE_DATA_TYPE ${DATA_TYPE}${VECTOR_COUNT})
else()
    set(DEVICE_DATA_TYPE ${DATA_TYPE})
endif()

set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "Used flags for the AOC compiler")
separate_arguments(AOC_FLAGS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../extern/hlslib/cmake)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

configure_file(
        "${CMAKE_SOURCE_DIR}/src/common/parameters.h.in"
        "${CMAKE_BINARY_DIR}/src/common/parameters.h"
)
include_directories(${CMAKE_BINARY_DIR}/src/common)

find_package(OpenMP)
find_package(IntelFPGAOpenCL)
find_package(Vitis)

add_subdirectory(src/device)
add_subdirectory(src/host)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()


