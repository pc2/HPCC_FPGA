# 'lib' is the folder with Google Test sources
add_subdirectory(../../extern/googletest ${CMAKE_CURRENT_BINARY_DIR}/lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR}/src/common ../../extern/cxxopts/include ../../shared/)


set(PROJECT_SOURCES ../../shared/setup/fpga_setup.cpp ../src/host/execution_blocked_pvt.cpp ../src/host/linpack_functionality.cpp)
set(TEST_SOURCES ../../shared/setup/test_fpga_setup.cpp test_kernel_functionality_and_host_integration.cpp test_kernel_functionality_separate_cores.cpp)

set(BLA_VENDOR Intel10_64lp)
find_package(LAPACK)

if (NOT LAPACK_FOUND)
    message(WARNING "MKL not found! Will not implement all unit tests since MKL is needed to run them.")
endif()


if (INTELFPGAOPENCL_FOUND)
    include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
    add_executable(Test_intel ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_intel gtest gmock gtest_main ${IntelFPGAOpenCL_LIBRARIES} "${OpenMP_CXX_FLAGS}")
    target_compile_definitions(Test_intel PRIVATE -DINTEL_FPGA)
    if (LAPACK_FOUND)
        target_compile_definitions(Test_intel PRIVATE -D_INTEL_MKL_)
        target_link_libraries(Test_intel ${LAPACK_LIBRARIES})
        include_directories(SYSTEM $ENV{MKLROOT}/include)
    endif()
    add_dependencies(Test_intel lu_blocked_pvt_emulate_intel lu_blocked_pvt_test_emulate_intel)
    target_compile_options(Test_intel PRIVATE "${OpenMP_CXX_FLAGS}")
    add_test(NAME test_intel_unit COMMAND $<TARGET_FILE:Test_intel> WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (Vitis_FOUND)
    include_directories(${Vitis_INCLUDE_DIRS})
    add_executable(Test_xilinx ${TEST_SOURCES} ${PROJECT_SOURCES})
    target_link_libraries(Test_xilinx gtest gmock gtest_main ${Vitis_LIBRARIES} "${OpenMP_CXX_FLAGS}")
    target_compile_definitions(Test_xilinx PRIVATE -DXILINX_FPGA)
    if (LAPACK_FOUND)
        target_compile_definitions(Test_xilinx PRIVATE -D_INTEL_MKL_)
        target_link_libraries(Test_xilinx ${LAPACK_LIBRARIES})
        include_directories(SYSTEM $ENV{MKLROOT}/include)
    endif()
    add_dependencies(Test_xilinx lu_blocked_pvt_emulate_xilinx)
    # Disabled since compilation is not possible
    #add_dependencies(Test_xilinx lu_blocked_pvt_test_emulate_xilinx)
    target_compile_options(Test_xilinx PRIVATE "${OpenMP_CXX_FLAGS}")
    add_test(NAME test_xilinx_unit COMMAND $<TARGET_FILE:Test_xilinx> WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
