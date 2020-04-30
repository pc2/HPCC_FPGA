
include_directories(../../../extern/cxxopts/include ../../../shared)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HOST_SOURCE execution_default.cpp main.cpp ../../../shared/setup/fpga_setup.cpp transpose_functionality.cpp)

add_executable(fTrans ${HOST_SOURCE} )
target_link_libraries(fTrans ${IntelFPGAOpenCL_LIBRARIES})
