project(HPCCBaseLibrary VERSION 1.0.1)


if (USE_ACCL)
add_library(hpcc_fpga_base STATIC ${CMAKE_CURRENT_SOURCE_DIR}/setup/fpga_setup.cpp ${CMAKE_CURRENT_SOURCE_DIR}/setup/fpga_setup_accl.cpp)
target_include_directories(hpcc_fpga_base PRIVATE ${ACCL_INCLUDE_PATH})
target_link_libraries(hpcc_fpga_base accl)
else()
add_library(hpcc_fpga_base STATIC ${CMAKE_CURRENT_SOURCE_DIR}/setup/fpga_setup.cpp)
endif()

find_package(OpenCL QUIET)

if (INTELFPGAOPENCL_FOUND)
    target_include_directories(hpcc_fpga_base PUBLIC ${IntelFPGAOpenCL_INCLUDE_DIRS})
elseif(Vitis_FOUND)
    target_include_directories(hpcc_fpga_base PUBLIC ${Vitis_INCLUDE_DIRS})  
elseif(OpenCL_FOUND)
    target_include_directories(hpcc_fpga_base PUBLIC ${OpenCL_INCLUDE_DIRS})  
else()
    message(ERROR "No OpenCL header found on system!")
endif()

target_include_directories(hpcc_fpga_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hpcc_fpga_base cxxopts)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
