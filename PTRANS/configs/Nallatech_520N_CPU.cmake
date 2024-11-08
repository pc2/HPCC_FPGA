# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)

set(CMAKE_CXX_FLAGS "-march=native" CACHE STRING "Additional flags sued for every build type" FORCE)
set(CMAKE_C_FLAGS "-march=native" CACHE STRING "Additional flags sued for every build type" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices" FORCE)
set(BLOCK_SIZE 16 CACHE STRING "Block size used in the FPGA kernel" FORCE)
set(NUM_REPLICATIONS 1 CACHE STRING "Number of kernel replications (should match number of external channels here)" FORCE)
