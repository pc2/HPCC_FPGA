
set(BLA_VENDOR Intel10_64lp)
find_package(LAPACK)

if (NOT LAPACK_FOUND)
    message(WARNING "MKL not found! Will not implement all unit tests since MKL is needed to run them.")
endif()

set(HOST_EXE_NAME Linpack)
set(LIB_NAME lp)

set(TEST_SOURCES test_kernel_functionality_and_host_integration.cpp test_kernel_functionality_separate_cores.cpp)

include(${CMAKE_SOURCE_DIR}/../cmake/unitTestTargets.cmake)

if (LAPACK_FOUND)
    if (INTELFPGAOPENCL_FOUND)
        target_compile_definitions(${HOST_EXE_NAME}_test_intel PRIVATE -D_INTEL_MKL_)
        target_link_libraries(${HOST_EXE_NAME}_test_intel ${LAPACK_LIBRARIES})
    endif()
    if (Vitis_FOUND)
        target_compile_definitions(${HOST_EXE_NAME}_test_xilinx PRIVATE -D_INTEL_MKL_)
        target_link_libraries(${HOST_EXE_NAME}_test_xilinx ${LAPACK_LIBRARIES})
    endif()
    include_directories(SYSTEM $ENV{MKLROOT}/include)
endif()



