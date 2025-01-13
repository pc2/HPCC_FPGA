# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(USE_ACCL No CACHE BOOL "" FORCE)
set(USE_UDP Yes CACHE BOOL "" FORCE)
set(USE_XRT_HOST Yes CACHE BOOL "" FORCE)
set(USE_OCL_HOST No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "xilinx_u280_gen3x16_xdma_1_202211_1" CACHE STRING "" FORCE)
set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.vnx.u280.hbm.ini CACHE FILEPATH "" FORCE)
set(XILINX_COMPILE_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.u280.ini CACHE FILEPATH "" FORCE)
set(XILINX_ADDITIONAL_LINK_FLAGS --kernel_frequency 250 CACHE STRING "" FORCE)
set(XILINX_KERNEL_NAMES "recv_stream" "send_stream" CACHE STRING "" FORCE)
# STREAM specific options
# Defaults to a total of ~12GB data
set(CHANNEL_WIDTH 0 CACHE STRING "Width of a single external channel in Byte will not be considered" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of kernel replications: One for each network stack" FORCE)
set(VNX_UDP_ETH_IFS 2 CACHE STRING "Number of ETH interfaces should match number of replications and number of available QSFP ports")

set(USE_DEPRECATED_HPP_HEADER Yes CACHE BOOL "Use cl.hpp intead of cl2.hpp" FORCE)
