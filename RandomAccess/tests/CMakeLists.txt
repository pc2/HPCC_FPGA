
set(HOST_EXE_NAME RandomAccess)
set(LIB_NAME ra)

set(TEST_SOURCES test_host_code.cpp test_kernel_functionality_and_host_integration.cpp)

include(${CMAKE_SOURCE_DIR}/../cmake/unitTestTargets.cmake)