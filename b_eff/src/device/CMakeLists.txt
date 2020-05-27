
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

generate_kernel_targets_intel(communication_bw520n communication_bw520n_disable_pipelining communication_bw520n_combined_loops)
add_test(NAME test_emulation_intel COMMAND ./Network_intel -f communication_bw520n_emulate.aocx -l 1 -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Network_intel -f communication_bw520n_emulate.aocx -l 1 -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
