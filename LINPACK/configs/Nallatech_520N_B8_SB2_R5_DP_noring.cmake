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

set(DATA_TYPE "double" CACHE STRING "The ued data type for calculation" FORCE)

# LINPACK specific options
set(DEFAULT_MATRIX_SIZE 1024 CACHE STRING "Default matrix size" FORCE)
set(LOCAL_MEM_BLOCK_LOG 8 CACHE STRING "Used to define the width and height of the block stored in local memory" FORCE)
set(REGISTER_BLOCK_LOG 2 CACHE STRING "Size of the block that will be manipulated in registers" FORCE)
set(NUM_REPLICATIONS 5 CACHE STRING "Number of times the matrix multiplication kernel will be replicated" FORCE)

