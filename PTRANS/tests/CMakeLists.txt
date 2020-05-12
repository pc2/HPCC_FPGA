
# 'lib' is the folder with Google Test sources
add_subdirectory(../../extern/googletest ${CMAKE_CURRENT_BINARY_DIR}/lib)
include_directories(${gtest_SOURCE_DIR}/include ${gmock_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/src/common ../../extern/cxxopts/include ../../shared)
include_directories(${CMAKE_SOURCE_DIR}/src/host)

set(PROJECT_SOURCES ../../shared/setup/fpga_setup.cpp ../src/host/common_benchmark_io_implementation.cpp ../src/host/execution_default.cpp ../src/host/transpose_functionality.cpp)
set(TEST_SOURCES ../../shared/testing/main.cpp test_host_functionality.cpp test_kernel_functionality_and_host_integration.cpp ../../shared/setup/test_fpga_setup.cpp)

if (INTELFPGAOPENCL_FOUND)
    include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
    add_executable(Test_intel ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_intel gtest gmock ${IntelFPGAOpenCL_LIBRARIES})
    add_dependencies(Test_intel transpose_optimized_emulate_intel)
    target_compile_definitions(Test_intel PRIVATE -DINTEL_FPGA)
    add_test(NAME test_intel_unit COMMAND $<TARGET_FILE:Test_intel> -f transpose_optimized_emulate.aocx WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
    include_directories(${Vitis_INCLUDE_DIRS})
    add_executable(Test_xilinx ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_xilinx gtest gmock ${Vitis_LIBRARIES})
    add_dependencies(Test_xilinx transpose_optimized_emulate_xilinx)
    target_compile_definitions(Test_xilinx PRIVATE -DXILINX_FPGA)
    add_test(NAME test_xilinx_unit COMMAND $<TARGET_FILE:Test_xilinx> -f transpose_optimized_emulate.xclbin WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()