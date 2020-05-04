
include_directories(../../../extern/cxxopts/include ../../../shared)
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HOST_SOURCE execution_default.cpp main.cpp ../../../shared/setup/fpga_setup.cpp transpose_functionality.cpp)

if (INTELFPGAOPENCL_FOUND)
    include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
    add_executable(trans_intel ${HOST_SOURCE} )
    target_link_libraries(trans_intel ${IntelFPGAOpenCL_LIBRARIES})
    target_compile_definitions(trans_intel PRIVATE -DINTEL_FPGA)
endif()

if (VITIS_FOUND)
    include_directories(${Vitis_INCLUDE_DIRS})
    add_executable(trans_xilinx ${HOST_SOURCE})
    target_link_libraries(trans_xilinx ${Vitis_LIBRARIES})
    target_compile_definitions(trans_xilinx PRIVATE -DXILINX_FPGA)
endif()
