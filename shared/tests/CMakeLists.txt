
set(HPCC_BASE_TEST_SOURCES  main.cpp hpcc_base_benchmark_test.cpp)

add_library(hpcc_fpga_base_test STATIC ${HPCC_BASE_TEST_SOURCES})
target_link_libraries(hpcc_fpga_base_test gtest gmock hpcc_fpga_base)
target_include_directories(hpcc_fpga_base_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (INTELFPGAOPENCL_FOUND)
    target_include_directories(hpcc_fpga_base_test PUBLIC ${IntelFPGAOpenCL_INCLUDE_DIRS})
    target_link_libraries(hpcc_fpga_base_test ${IntelFPGAOpenCL_LIBRARIES})
elseif(Vitis_FOUND)
    target_include_directories(hpcc_fpga_base_test PUBLIC ${Vitis_INCLUDE_DIRS})  
    target_link_libraries(hpcc_fpga_base_test ${Vitis_LIBRARIES})
else()
    message(ERROR "No OpenCL header found on system!")
endif()
