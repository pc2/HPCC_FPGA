
# 'lib' is the folder with Google Test sources
add_subdirectory(../../extern/googletest ${CMAKE_CURRENT_BINARY_DIR}/lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/src/common ../../extern/cxxopts/include ../../shared)
include_directories(SYSTEM ${IntelFPGAOpenCL_INCLUDE_DIRS})
include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${CMAKE_SOURCE_DIR}/src/host)

set(PROJECT_SOURCES ../../shared/setup/fpga_setup.cpp ../src/host/common_benchmark_io_implementation.cpp ../src/host/execution_default.cpp)
set(TEST_SOURCES ../../shared/testing/main.cpp ../../shared/setup/test_fpga_setup.cpp test_kernel_functionality_and_host_integration.cpp)

add_executable(Google_Tests_run ${TEST_SOURCES} ${PROJECT_SOURCES})
target_link_libraries(Google_Tests_run gtest gmock ${IntelFPGAOpenCL_LIBRARIES} ${MPI_LIBRARIES})
add_dependencies(Google_Tests_run communication_bw520n_emulate_intel)
target_compile_options(Google_Tests_run PRIVATE -D_USE_MPI_)
add_test(NAME test_intel_unit COMMAND $<TARGET_FILE:Google_Tests_run> -f communication_bw520n_emulate.aocx WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})