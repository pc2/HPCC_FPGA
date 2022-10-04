# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "xilinx_u55c_gen3x16_xdma_3_202210_1" CACHE STRING "" FORCE)
set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.transpose_pq_pcie.hbm.generator.ini CACHE FILEPATH "" FORCE)
set(XILINX_COMPILE_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.transpose_pq_pcie.hbm.ini CACHE FILEPATH "" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices" FORCE)
set(BLOCK_SIZE 256 CACHE STRING "Block size used in the FPGA kernel" FORCE)
set(CHANNEL_WIDTH 16 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of kernel replications (should match number of external channels here)" FORCE)

set(USE_DEPRECATED_HPP_HEADER Yes CACHE BOOL "Use cl.hpp intead of cl2.hpp" FORCE)
