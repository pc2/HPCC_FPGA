# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "xilinx_u250_xdma_201830_2" CACHE STRING "" FORCE)
set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.transpose_pq_pcie.u250.generator.ini CACHE FILEPATH "" FORCE)
set(XILINX_COMPILE_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.transpose_pq_pcie.u250.ini CACHE FILEPATH "" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices" FORCE)
set(BLOCK_SIZE 64 CACHE STRING "Block size used in the FPGA kernel" FORCE)
set(CHANNEL_WIDTH 16 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory" FORCE)
set(NUM_REPLICATIONS 4 CACHE STRING "Number of kernel replications (should match number of external channels here)" FORCE)
set(XILINX_UNROLL_INNER_LOOPS Yes CACHE BOOL "When building for Xilinx devices, unroll the inner loops to create a single pipeline per block and keep memory bursts. This is a tradeoff between resource usage and performance." FORCE)

set(USE_DEPRECATED_HPP_HEADER Yes CACHE BOOL "Use cl.hpp intead of cl2.hpp" FORCE)
