# This file contains the default configuration for the Intel PAC D5005 board
# using USM for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "xilinx_u280_xdma_201920_3" CACHE STRING "" FORCE)
set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.random_access_kernels_single.ddr.ini CACHE FILEPATH "" FORCE)
set(XILINX_COMPILE_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.random_access_kernels_single.ddr.ini CACHE FILEPATH "" FORCE)


# RA specific options
set(HPCC_FPGA_RA_DEFAULT_ARRAY_LENGTH_LOG 29 CACHE STRING "" FORCE)
set(HPCC_FPGA_RA_INTEL_USE_PRAGMA_IVDEP No CACHE BOOL "" FORCE)
set(HPCC_FPGA_RA_DEVICE_BUFFER_SIZE_LOG 10 CACHE STRING "" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "" FORCE)
set(HPCC_FPGA_RA_GLOBAL_MEM_UNROLL_LOG 1 CACHE STRING "" FORCE)
set(HPCC_FPGA_RA_RNG_COUNT_LOG 1 CACHE BOOL "Log2 of the number of random number generators that will be used concurrently" FORCE)
set(HPCC_FPGA_RA_RNG_DISTANCE 1 CACHE BOOL "Distance between RNGs in shift register. Used to relax data dependencies and increase clock frequency" FORCE)
