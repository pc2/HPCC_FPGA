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
set(FPGA_BOARD_NAME "xilinx_u55c_gen3x16_xdma_3_202210_1" CACHE STRING "" FORCE)
set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.accl_pl.u55c.hbm.ini CACHE FILEPATH "" FORCE)
set(XILINX_COMPILE_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.accl_buffers.ini CACHE FILEPATH "" FORCE)
set(XILINX_ADDITIONAL_LINK_FLAGS --kernel_frequency 250 CACHE STRING "" FORCE)
set(XILINX_KERNEL_NAMES "send_recv" CACHE STRING "" FORCE)
set(USE_ACCL_CLIENT_ARBITER Yes CACHE BOOL "Use the client arbiter kernel to connect multiple kernels to the CCLO cmd stream" FORCE)
# STREAM specific options
# Defaults to a total of ~12GB data
set(CHANNEL_WIDTH 0 CACHE STRING "Width of a single external channel in Byte will not be considered" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of kernel replications will not be considered" FORCE)

set(USE_DEPRECATED_HPP_HEADER Yes CACHE BOOL "Use cl.hpp intead of cl2.hpp" FORCE)
