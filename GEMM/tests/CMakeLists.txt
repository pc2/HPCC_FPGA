
# 'lib' is the folder with Google Test sources
add_subdirectory(../../extern/googletest ${CMAKE_CURRENT_BINARY_DIR}/lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/src/common ../../extern/cxxopts/include ../../shared/)
include_directories(${CMAKE_SOURCE_DIR}/src/host)

set(PROJECT_SOURCES ../../shared/setup/fpga_setup.cpp ../src/host/common_benchmark_io_implementation.cpp ../src/host/gemm_functionality.cpp ../src/host/execution_cannon.cpp)
set(TEST_SOURCES ../../shared/testing/main.cpp ../../shared/setup/test_fpga_setup.cpp test_kernel_functionality_and_host_integration.cpp)

if (INTELFPGAOPENCL_FOUND)
    include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
    add_executable(Test_intel ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_intel gtest gmock ${IntelFPGAOpenCL_LIBRARIES})
    target_compile_definitions(GEMM_intel PRIVATE -DINTEL_FPGA)
    add_dependencies(Test_intel gemm_cannon_emulate_intel)
    add_test(NAME test_intel_unit COMMAND $<TARGET_FILE:Test_intel> -f gemm_cannon_emulate.aocx WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (Vitis_FOUND)
    include_directories(${Vitis_INCLUDE_DIRS})
    add_executable(Test_xilinx ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_xilinx gtest gmock ${Vitis_LIBRARIES})
    target_compile_definitions(Test_xilinx PRIVATE -DXILINX_FPGA)
    add_dependencies(Test_xilinx gemm_cannon_emulate_xilinx)  
    add_test(NAME test_xilinx_unit COMMAND $<TARGET_FILE:Test_xilinx> -f gemm_cannon_emulate.xclbin WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
