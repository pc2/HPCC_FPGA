
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

math(EXPR TEST_MATRIX_SIZE "1 << (${LOCAL_MEM_BLOCK_LOG} + 1)")

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(hpl_torus)
    add_test(NAME test_emulation_intel COMMAND Linpack_intel -f hpl_torus_emulate.aocx -s ${TEST_MATRIX_SIZE} -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Linpack_intel ${TEST_HOST_FLAGS} -f hpl_torus_emulate.aocx -s ${TEST_MATRIX_SIZE} -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(hpl_torus)
    add_test(NAME test_emulation_xilinx COMMAND Linpack_xilinx -f hpl_torus_emulate.xclbin -s ${TEST_MATRIX_SIZE} -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Linpack_xilinx ${TEST_HOST_FLAGS} -f hpl_torus_emulate.xclbin -s ${TEST_MATRIX_SIZE} -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

