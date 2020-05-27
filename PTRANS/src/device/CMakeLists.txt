
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(transpose_optimized)
    add_test(NAME test_emulation_optimized_intel COMMAND Transpose_intel -f transpose_optimized_emulate.aocx -n 1 -m 1 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Transpose_intel -f transpose_optimized_emulate.aocx -n 1 -m 1
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(transpose_optimized)
    add_test(NAME test_emulation_optimized_xilinx COMMAND Transpose_xilinx -f transpose_optimized_emulate.xclbin -n 1 -m 1 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Transpose_xilinx -f transpose_optimized_emulate.xclbin -n 1 -m 1
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

