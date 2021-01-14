# This file contains the default configuration for the Intel PAC D5005 board
# using USM for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI No CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM Yes CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_hpc_m210h" CACHE STRING "" FORCE)
set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "" FORCE)

# RA specific options
set(DEFAULT_ARRAY_LENGTH 536870912 CACHE STRING "" FORCE)
set(INTEL_USE_PRAGMA_IVDEP No CACHE BOOL "" FORCE)
set(DEVICE_BUFFER_SIZE 1024 CACHE STRING "" FORCE)
set(NUM_REPLICATIONS 32 CACHE STRING "" FORCE)
set(INTEL_CODE_GENERATION_SETTINGS ${CMAKE_SOURCE_DIR}/settings/settings.gen.intel.random_access_kernels_single.hbm.py CACHE FILEPATH "" FORCE)
