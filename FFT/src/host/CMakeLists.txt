
include_directories(../../../extern/cxxopts/include ../../../shared)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HOST_SOURCE execution_default.cpp main.cpp ../../../shared/setup/fpga_setup.cpp fft_functionality.cpp)

add_executable(fFFT ${HOST_SOURCE})
target_link_libraries(fFFT ${IntelFPGAOpenCL_LIBRARIES})
