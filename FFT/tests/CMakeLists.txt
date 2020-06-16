
set(HOST_EXE_NAME FFT)
set(LIB_NAME fft_lib)

set(TEST_SOURCES test_fft_functionality.cpp test_execution_functionality.cpp)

include(${CMAKE_SOURCE_DIR}/../cmake/unitTestTargets.cmake)
