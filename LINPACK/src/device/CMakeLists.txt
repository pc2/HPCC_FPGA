
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(hpl_torus_IEC hpl_torus_PCIE)
    add_test(NAME test_emulation_intel COMMAND Linpack_intel -f hpl_torus_emulate.aocx -m 2 -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Linpack_intel ${TEST_HOST_FLAGS} -f hpl_torus_emulate.aocx -m 2 -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(hpl_torus_PCIE)
    add_test(NAME test_emulation_xilinx COMMAND Linpack_xilinx -f hpl_torus_emulate.xclbin -m 2 -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Linpack_xilinx ${TEST_HOST_FLAGS} -f hpl_torus_emulate.xclbin -m 2 -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

