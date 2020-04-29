
add_compile_options(--std=c++11)
# 'lib' is the folder with Google Test sources
add_subdirectory(../../extern/googletest ${CMAKE_CURRENT_BINARY_DIR}/lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/src/common ../../extern/cxxopts/include ../../shared)
include_directories(SYSTEM ${IntelFPGAOpenCL_INCLUDE_DIRS})

set(PROJECT_SOURCES ../../shared/setup/fpga_setup.cpp ../src/host/execution_default.cpp ../src/host/fft_functionality.cpp)
set(TEST_SOURCES ../../shared/setup/test_fpga_setup.cpp "test_fft_functionality.cpp" test_execution_functionality.cpp)

add_executable(Google_Tests_run ${TEST_SOURCES} ${PROJECT_SOURCES})
target_link_libraries(Google_Tests_run gtest gmock gtest_main ${IntelFPGAOpenCL_LIBRARIES})
add_dependencies(Google_Tests_run fft1d_float_8_emulate)
add_test(NAME test_intel_unit COMMAND $<TARGET_FILE:Google_Tests_run> WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})