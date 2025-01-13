
set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "xilinx_u280_gen3x16_xdma_1_202211_1" CACHE STRING "" FORCE)
set(XILINX_LINK_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.u280.hbm.ini CACHE FILEPATH "" FORCE)
set(XILINX_COMPILE_SETTINGS_FILE ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.u280.ini CACHE FILEPATH "" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(CHANNEL_WIDTH 0 CACHE STRING "Width of a single external channel in Byte will not be considered" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of kernel replications will not be considered" FORCE)

set(USE_DEPRECATED_HPP_HEADER Yes CACHE BOOL "Use cl.hpp intead of cl2.hpp" FORCE)
