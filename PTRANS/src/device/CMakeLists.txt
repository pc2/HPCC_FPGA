
set(KERNEL_REPLICATION_ENABLED Yes CACHE INTERNAL "Enables kernel replication in the CMake target generation function")

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(transpose_DIAG_IEC transpose_c2_DIAG_IEC transpose_PQ_PCIE transpose_DIAG_PCIE transpose_PQ_IEC)
    add_test(NAME test_emulation_diagonal_intel COMMAND Transpose_intel -f transpose_DIAG_IEC_emulate.aocx -n 1 -m 1 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Transpose_intel -f transpose_DIAG_IEC_emulate.aocx -n 1 -m 1
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(transpose_PQ_PCIE transpose_DIAG_PCIE transpose_PQ_ACCL_buffers transpose_PQ_ACCL_stream transpose_PQ_ACCL_stream_sendrecv)
    add_test(NAME test_emulation_PQ_PCIE_xilinx COMMAND Transpose_xilinx -f transpose_PQ_PCIE_emulate.xclbin -n 1 -m 1 WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Transpose_xilinx -f transpose_PQ_PCIE_emulate.xclbin -n 1 -m 1
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

