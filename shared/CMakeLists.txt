project(HPCCBaseLibrary VERSION 1.0.0)

add_library(hpcc_fpga_base STATIC ${CMAKE_CURRENT_SOURCE_DIR}/hpcc_benchmark.cpp ${CMAKE_CURRENT_SOURCE_DIR}/setup/fpga_setup.cpp)

target_include_directories(hpcc_fpga_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../extern/cxxopts/include)

install(TARGETS hpcc_fpga_base 
PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")