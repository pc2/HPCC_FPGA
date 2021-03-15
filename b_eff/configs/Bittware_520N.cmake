# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_max_sg280l" CACHE STRING "" FORCE)
set(AOC_FLAGS "-fpc -fp-relaxed -seed=7" CACHE STRING "" FORCE)

# GEMM specific options
set(CHANNEL_WIDTH 32 CACHE STRING "Width of a single external channel in Byte" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of kernel replications" FORCE)