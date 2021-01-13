
set(HOST_EXE_NAME Transpose)
set(LIB_NAME trans)

set(TEST_SOURCES test_host_functionality.cpp test_kernel_functionality_and_host_integration.cpp test_transpose_data_handlers.cpp)

include(${CMAKE_SOURCE_DIR}/../cmake/unitTestTargets.cmake)
