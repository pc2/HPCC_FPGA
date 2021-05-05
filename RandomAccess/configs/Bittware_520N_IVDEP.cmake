# This file contains the default configuration for the Intel PAC D5005 board
# using USM for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_hpc_sg280l" CACHE STRING "" FORCE)
set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "" FORCE)

# RA specific options
set(HPCC_FPGA_RA_DEFAULT_ARRAY_LENGTH_LOG 29 CACHE STRING "" FORCE)
set(HPCC_FPGA_RA_INTEL_USE_PRAGMA_IVDEP Yes CACHE BOOL "" FORCE)
set(HPCC_FPGA_RA_DEVICE_BUFFER_SIZE_LOG 0 CACHE STRING "" FORCE)
set(HPCC_FPGA_RA_NUM_REPLICATIONS 4 CACHE STRING "" FORCE)
set(HPCC_FPGA_RA_RNG_COUNT_LOG 6 CACHE BOOL "Log2 of the number of random number generators that will be used concurrently" FORCE)
set(HPCC_FPGA_RA_RNG_DISTANCE 5 CACHE BOOL "Distance between RNGs in shift register. Used to relax data dependencies and increase clock frequency" FORCE)
