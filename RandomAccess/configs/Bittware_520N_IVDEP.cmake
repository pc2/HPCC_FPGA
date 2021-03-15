# This file contains the default configuration for the Intel PAC D5005 board
# using USM for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI No CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_hpc_sg280l" CACHE STRING "" FORCE)
set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "" FORCE)

# RA specific options
set(DEFAULT_ARRAY_LENGTH 536870912 CACHE STRING "" FORCE)
set(INTEL_USE_PRAGMA_IVDEP Yes CACHE BOOL "" FORCE)
set(DEVICE_BUFFER_SIZE 1 CACHE STRING "" FORCE)
set(NUM_REPLICATIONS 4 CACHE STRING "" FORCE)
