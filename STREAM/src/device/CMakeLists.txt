
set(KERNEL_REPLICATION_ENABLED Yes)

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(stream_kernels stream_kernels_single)
    add_test(NAME test_emulation_intel COMMAND STREAM_FPGA_intel -f stream_kernels_emulate.aocx -n 1
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_single_emulation_intel COMMAND STREAM_FPGA_intel -f stream_kernels_single_emulate.aocx --single-kernel -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./STREAM_FPGA_intel -f stream_kernels_single_emulate.aocx --single-kernel -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(stream_kernels stream_kernels_single)
    add_test(NAME test_single_emulation_xilinx COMMAND STREAM_FPGA_xilinx -f stream_kernels_single_emulate.xclbin --single-kernel -n 1
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./STREAM_FPGA_xilinx -f stream_kernels_single_emulate.xclbin --single-kernel -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
