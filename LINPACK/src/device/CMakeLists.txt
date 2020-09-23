
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(lu_blocked_pvt lu_blocked_pvt_test)
    add_test(NAME test_emulation_intel COMMAND Linpack_intel -f lu_blocked_pvt_emulate.aocx -s 128 -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Linpack_intel ${TEST_HOST_FLAGS} -f lu_blocked_pvt_emulate.aocx -s 128 -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(lu_blocked_pvt)
    add_test(NAME test_emulation_xilinx COMMAND Linpack_xilinx -f lu_blocked_pvt_emulate.xclbin -s 128 -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Linpack_xilinx ${TEST_HOST_FLAGS} -f lu_blocked_pvt_emulate.xclbin -s 128 -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

