
set(HOST_EXE_NAME Network)
set(LIB_NAME net_lib)

set(TEST_SOURCES test_kernel_functionality_and_host_integration.cpp)

include(${CMAKE_SOURCE_DIR}/../cmake/unitTestTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    target_link_libraries(${LIB_NAME}_intel ${MPI_LIBRARIES})
endif()
if (Vitis_FOUND)
    target_link_libraries(${LIB_NAME}_xilinx ${MPI_LIBRARIES})
endif()