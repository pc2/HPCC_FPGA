
set(TEST_SOURCES test_kernel_functionality_and_host_integration.cpp)

set(HOST_EXE_NAME STREAM_FPGA)
set(LIB_NAME stream)

include(${CMAKE_SOURCE_DIR}/../cmake/unitTestTargets.cmake)