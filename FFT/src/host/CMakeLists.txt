
include_directories(../../../extern/cxxopts/include ../../../shared)
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HOST_SOURCE execution_default.cpp main.cpp ../../../shared/setup/fpga_setup.cpp fft_functionality.cpp)

if (INTELFPGAOPENCL_FOUND)
    include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
    add_executable(FFT_intel ${HOST_SOURCE})
    target_compile_definitions(FFT_intel PRIVATE -DINTEL_FPGA)
    target_link_libraries(FFT_intel ${IntelFPGAOpenCL_LIBRARIES})
endif()

if (Vitis_FOUND)
    include_directories(${Vitis_INCLUDE_DIRS})
    add_executable(FFT_xilinx ${HOST_SOURCE})
    target_compile_definitions(FFT_xilinx PRIVATE -DXILINX_FPGA)
    target_link_libraries(FFT_xilinx ${Vitis_LIBRARIES})
endif()
