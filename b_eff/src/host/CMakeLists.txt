
include_directories(../../../extern/cxxopts/include ../../../shared)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}/src/common)
include_directories(${CMAKE_SOURCE_DIR}/../shared/setup .)

set(HOST_SOURCE execution_default.cpp main.cpp common_benchmark_io_implementation.cpp ../../../shared/setup/fpga_setup.cpp network_functionality.cpp)

add_executable(fnet ${HOST_SOURCE})
target_link_libraries(fnet ${IntelFPGAOpenCL_LIBRARIES} ${MPI_LIBRARIES})
target_compile_options(fnet PRIVATE -D_USE_MPI_)
