
include_directories(../../../extern/cxxopts/include)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HEADER_FILES execution.h setup/fpga_setup.hpp network_functionality.hpp)
set(HOST_SOURCE execution_default.cpp main.cpp setup/fpga_setup.cpp network_functionality.cpp)

add_compile_options(--std=c++11)

add_executable(fnet ${HOST_SOURCE} ${HEADER_FILES})
target_link_libraries(fnet ${IntelFPGAOpenCL_LIBRARIES} ${MPI_LIBRARIES})
