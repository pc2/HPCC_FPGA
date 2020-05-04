
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

generate_kernel_targets_intel(fft1d_float_8)
add_test(NAME test_emulation_intel COMMAND ./fFFT -f fft1d_float_8_emulate.aocx -i 1 -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./fFFT -f fft1d_float_8_emulate.aocx -i 1 -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
