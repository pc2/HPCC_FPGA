
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

generate_kernel_targets_intel(transpose_default transpose_optimized)
add_test(NAME test_emulation_optimized_intel COMMAND fTrans -f transpose_optimized_emulate.aocx -m ${BLOCK_SIZE} WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
add_test(NAME test_emulation_default_intel COMMAND fTrans -f transpose_default_emulate.aocx -m ${BLOCK_SIZE} WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
