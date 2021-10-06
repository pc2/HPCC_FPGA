# This file contains the default configuration for the Nallatech 520N board
# for the use with single precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI Yes CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "xilinx_u280_xdma_201920_3" CACHE STRING "" FORCE)

# LINPACK specific options
set(DEFAULT_MATRIX_SIZE 1024 CACHE STRING "Default matrix size" FORCE)
set(LOCAL_MEM_BLOCK_LOG 8 CACHE STRING "Used to define the width and height of the block stored in local memory" FORCE)
set(REGISTER_BLOCK_LOG 3 CACHE STRING "Size of the block that will be manipulated in registers" FORCE)
set(NUM_REPLICATIONS 2 CACHE STRING "Number of times the matrix multiplication kernel will be replicated" FORCE)

set(XILINX_COMPILE_SETTINGS_FILE "${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx.hpl_torus_pcie.ddr.ini" CACHE STRING "Compile settings file" FORCE)
set(XILINX_LINK_SETTINGS_FILE "${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx.hpl_torus_pcie.ddr.generator.ini" CACHE STRING "Link settings file" FORCE)

set(USE_DEPRECATED_HPP_HEADER Yes CACHE BOOL "Use cl.hpp instead of cl2.hpp" FORCE)
set(USE_PCIE_MPI_COMMUNICATION Yes CACHE BOOL "Use PCIe and MPI for communication between FPGAs" FORCE)

