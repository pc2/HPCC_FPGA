set(KERNEL_REPLICATION_ENABLED Yes CACHE INTERNAL "Enables kernel replication in the CMake target genertion function")

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(fft1d_float_8)
    add_test(NAME test_emulation_intel COMMAND ./FFT_intel -f fft1d_float_8_emulate.aocx -b ${NUM_REPLICATIONS} -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./FFT_intel -f fft1d_float_8_emulate.aocx -b ${NUM_REPLICATIONS} -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (Vitis_FOUND)
    generate_kernel_targets_xilinx(fft1d_float_8)
    add_test(NAME test_emulation_xilinx COMMAND ./FFT_xilinx -f fft1d_float_8_emulate.xclbin -b ${NUM_REPLICATIONS} -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./FFT_xilinx -f fft1d_float_8_emulate.xclbin -b ${NUM_REPLICATIONS} -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
