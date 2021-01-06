# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI No CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM Yes CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_hpc_m210h" CACHE STRING "" FORCE)
set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "" FORCE)

# FFT specific options
set(LOG_FFT_SIZE 7 CACHE STRING "Log2 of the used FFT size" FORCE)
set(NUM_REPLICATIONS 16 CACHE STRING "Number of kernel replications" FORCE)
set(INTEL_CODE_GENERATION_SETTINGS ${CMAKE_SOURCE_DIR}/settings/settings.gen.intel.fft1d_float_8.hbm.py CACHE FILEPATH "" FORCE)
