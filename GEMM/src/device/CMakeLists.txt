set(KERNEL_REPLICATION_ENABLED Yes CACHE INTERNAL "Enables kernel replication in the CMake target generation function")

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(gemm_base)
    add_test(NAME test_emulation_intel COMMAND ./GEMM_intel -f gemm_base_emulate.aocx -n 1 -m ${NUM_REPLICATIONS}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./GEMM_intel -f gemm_base_emulate.aocx -n 1 -m 1 
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (Vitis_FOUND)
    generate_kernel_targets_xilinx(gemm_base)
    add_test(NAME test_emulation_xilinx COMMAND ./GEMM_xilinx -f gemm_base_emulate.xclbin -n 1 -m ${NUM_REPLICATIONS}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./GEMM_xilinx -f gemm_base_emulate.xclbin -n 1 -m 1 
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
