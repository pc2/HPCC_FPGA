
add_subdirectory(../../extern/googletest ${CMAKE_BINARY_DIR}/lib/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HOST_EXE_NAME GEMM)
set(LIB_NAME ge)

set(TEST_SOURCES main.cpp test_kernel_functionality_and_host_integration.cpp)

if (INTELFPGAOPENCL_FOUND)
    include_directories(SYSTEM ${IntelFPGAOpenCL_INCLUDE_DIRS})
    add_executable(${HOST_EXE_NAME}_test_intel ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(${HOST_EXE_NAME}_test_intel gtest gmock ${LIB_NAME}_intel ${IntelFPGAOpenCL_LIBRARIES} "${OpenMP_CXX_FLAGS}")
    add_dependencies(${HOST_EXE_NAME}_test_intel gemm_base_emulate_intel)
    target_compile_definitions(${HOST_EXE_NAME}_test_intel PRIVATE -DINTEL_FPGA)
    target_compile_options(${HOST_EXE_NAME}_test_intel PRIVATE "${OpenMP_CXX_FLAGS}")
    add_test(NAME ${HOST_EXE_NAME}_test_intel_unit COMMAND $<TARGET_FILE:${HOST_EXE_NAME}_test_intel> -f gemm_base_emulate.aocx WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (Vitis_FOUND)
    include_directories(SYSTEM ${Vitis_INCLUDE_DIRS})
    add_executable(${HOST_EXE_NAME}_test_xilinx ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(${HOST_EXE_NAME}_test_xilinx gtest gmock ${LIB_NAME}_xilinx ${Vitis_LIBRARIES} "${OpenMP_CXX_FLAGS}")

    add_dependencies(${HOST_EXE_NAME}_test_xilinx gemm_base_emulate_xilinx)
    target_compile_definitions(${HOST_EXE_NAME}_test_xilinx PRIVATE -DXILINX_FPGA)
    target_compile_options(${HOST_EXE_NAME}_test_xilinx PRIVATE "${OpenMP_CXX_FLAGS}")
    add_test(NAME ${HOST_EXE_NAME}_test_xilinx_unit COMMAND $<TARGET_FILE:${HOST_EXE_NAME}_test_xilinx> -f gemm_base_emulate.xclbin WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
