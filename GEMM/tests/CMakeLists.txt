
set(HOST_EXE_NAME GEMM)
set(LIB_NAME ge)

set(TEST_SOURCES test_kernel_functionality_and_host_integration.cpp)

include(${CMAKE_SOURCE_DIR}/../cmake/unitTestTargets.cmake)
