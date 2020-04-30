

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

generate_kernel_targets_intel(gemm_cannon)
add_test(NAME test_emulation_intel COMMAND ./fgemm -f gemm_cannon_emulate.aocx -n 1 -m ${BLOCK_SIZE}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
