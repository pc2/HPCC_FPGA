
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(lu_blocked_pvt lu_blocked_pvt_test)
    add_test(NAME test_emulation_intel COMMAND LINPACK_intel -f lu_blocked_pvt_emulate.aocx -s 128 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

