
include_directories(../../../extern/cxxopts/include)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HEADER_FILES execution.h setup/fpga_setup.hpp fft_functionality.hpp)
set(HOST_SOURCE execution_default.cpp main.cpp setup/fpga_setup.cpp fft_functionality.cpp)

add_compile_options(--std=c++11)

add_executable(fFFT ${HOST_SOURCE} ${HEADER_FILES})
target_link_libraries(fFFT ${IntelFPGAOpenCL_LIBRARIES})
