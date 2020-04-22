cmake_minimum_required(VERSION 3.1)
project(LINPACK VERSION 2.0.0)

set (CMAKE_CXX_STANDARD 11)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

set(DEFAULT_REPETITIONS 10 CACHE STRING "Default number of repetitions")
set(DATA_TYPE float)
set(DEFAULT_MATRIX_SIZE 1024 CACHE STRING "Default matrix size")
set(DEFAULT_DEVICE -1 CACHE STRING "Index of the default device to use")
set(DEFAULT_PLATFORM -1 CACHE STRING "Index of the default platform to use")
set(GLOBAL_MEM_UNROLL 16 CACHE STRING "Unrolling factor that is used for all loops in the kernels")
set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Name of the target FPGA board")
set(LOCAL_MEM_BLOCK_LOG 5 CACHE STRING "Used to define the width and height of the block stored in local memory")
set(REGISTER_BLOCK_LOG 3 CACHE STRING "Size of the block that will be manipulated in registers")
set(XILINX_GENERATE_LINK_SETTINGS Yes CACHE BOOL "Generate the link settings depending on the number of replications for the kernels")
set(XILINX_LINK_SETTINGS_FILE "${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.stream_kernels.ddr.ini" CACHE STRING "The link settings file that should be used when generating is disabled")
set(XILINX_COMPILE_FLAGS "-j 40" CACHE STRING "Additional compile flags for the v++ compiler")
set(XILINX_COMPILE_SETTINGS "${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.ini" CACHE STRING "Compile settings file for the v++ compiler")
separate_arguments(XILINX_COMPILE_FLAGS)

set(HOST_DATA_TYPE cl_${DATA_TYPE})
set(DEVICE_DATA_TYPE ${DATA_TYPE})

if (DATA_TYPE STREQUAL "double")
    set(_DP Yes)
    message(STATUS "Set DP flag since data type seems to be double precision")
else()
    set(_DP No)
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


