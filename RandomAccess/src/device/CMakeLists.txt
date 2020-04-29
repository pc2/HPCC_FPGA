
set(KERNEL_REPLICATION_ENABLED Yes)

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)


if (INTELFPGAOPENCL_FOUND)
generate_kernel_targets_intel(random_access_kernels_single)
add_test(NAME test_emulation_intel COMMAND RandomAccess_intel -f random_access_kernels_single_emulate.aocx -d 1048576
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
generate_kernel_targets_xilinx(random_access_kernels_single)
add_test(NAME test_emulation_xilinx COMMAND RandomAccess_xilinx -f random_access_kernels_single_emulate.xclbin -d 1048576
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
