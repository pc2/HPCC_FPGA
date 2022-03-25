# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_hpc_m210h_g3x16" CACHE STRING "" FORCE)
set(AOC_FLAGS "-ffp-contract=fast -ffp-reassociate -no-interleaving=default" CACHE STRING "" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices" FORCE)
set(BLOCK_SIZE 64 CACHE STRING "Block size used in the FPGA kernel" FORCE)
set(CHANNEL_WIDTH 8 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory" FORCE)
set(NUM_REPLICATIONS 32 CACHE STRING "Number of kernel replications (should match number of external channels here)" FORCE)
set(INTEL_CODE_GENERATION_SETTINGS ${CMAKE_SOURCE_DIR}/settings/settings.gen.intel.transpose_pq.s10mxhbm.py CACHE FILEPATH "" FORCE)
