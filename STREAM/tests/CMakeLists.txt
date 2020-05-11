
# 'lib' is the folder with Google Test sources
add_subdirectory(../../extern/googletest ${CMAKE_CURRENT_BINARY_DIR}/lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/src/common ../../extern/cxxopts/include ../../shared)
include_directories(${CMAKE_SOURCE_DIR}/src/host/)


set(PROJECT_SOURCES  ../../shared/setup/fpga_setup.cpp ../src/host/execution_default.cpp ../src/host/common_benchmark_io_implementation.cpp ../src/host/stream_functionality.cpp)
set(TEST_SOURCES ../../shared/setup/test_fpga_setup.cpp test_kernel_functionality_and_host_integration.cpp ../../shared/testing/main.cpp)

if (INTELFPGAOPENCL_FOUND)
    include_directories(SYSTEM ${IntelFPGAOpenCL_INCLUDE_DIRS})
    add_executable(Test_intel ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_intel gtest gmock ${IntelFPGAOpenCL_LIBRARIES} "${OpenMP_CXX_FLAGS}")
    add_dependencies(Test_intel stream_kernels_emulate_intel stream_kernels_single_emulate_intel)
    target_compile_definitions(Test_intel PRIVATE -DINTEL_FPGA)
    target_compile_options(Test_intel PRIVATE "${OpenMP_CXX_FLAGS}")
    add_test(NAME test_intel_unit COMMAND $<TARGET_FILE:Test_intel> -f stream_kernels_emulate.aocx WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_intel_single_unit COMMAND $<TARGET_FILE:Test_intel> -f stream_kernels_single_emulate.aocx  --single-kernel WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (Vitis_FOUND)
    include_directories(SYSTEM ${Vitis_INCLUDE_DIRS})
    add_executable(Test_xilinx ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_xilinx gtest gmock ${Vitis_LIBRARIES} "${OpenMP_CXX_FLAGS}")
    add_dependencies(Test_xilinx stream_kernels_single_emulate_xilinx)
    target_compile_definitions(Test_xilinx PRIVATE -DXILINX_FPGA)
    target_compile_options(Test_xilinx PRIVATE "${OpenMP_CXX_FLAGS}")
    add_test(NAME test_xilinx_single_unit COMMAND $<TARGET_FILE:Test_xilinx> -f stream_kernels_single_emulate.xclbin --single-kernel WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()