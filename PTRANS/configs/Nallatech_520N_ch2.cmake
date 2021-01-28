# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#
# This configuration is for the kernel transpose_diagonal_c2, which is using 2 external channels per kernel replication instead of 1


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_max_sg280l" CACHE STRING "" FORCE)
set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices" FORCE)
set(BLOCK_SIZE 512 CACHE STRING "Block size used in the FPGA kernel" FORCE)
set(CHANNEL_WIDTH 16 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of kernel replications (should match number of external channels here)" FORCE)
