# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(USE_ACCL Yes CACHE BOOL "" FORCE)
set(USE_XRT_HOST Yes CACHE BOOL "" FORCE)
set(USE_OCL_HOST No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "xilinx_u280_xdma_201920_3" CACHE STRING "" FORCE)
set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.transpose_pq_accl_buffers.ddr.ini CACHE FILEPATH "" FORCE)
set(XILINX_COMPILE_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.transpose_pq_pcie.ddr.ini CACHE FILEPATH "" FORCE)
set(XILINX_ADDITIONAL_LINK_FLAGS --kernel_frequency 250 CACHE STRING "" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(DEFAULT_MATRIX_SIZE 8 CACHE STRING "Default size of the used matrices" FORCE)
set(BLOCK_SIZE 256 CACHE STRING "Block size used in the FPGA kernel" FORCE)
set(CHANNEL_WIDTH 16 CACHE STRING "Width of a single channel in number of values. Also specifies the width of memory" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of kernel replications (should match number of external channels here)" FORCE)

set(USE_DEPRECATED_HPP_HEADER Yes CACHE BOOL "Use cl.hpp intead of cl2.hpp" FORCE)
