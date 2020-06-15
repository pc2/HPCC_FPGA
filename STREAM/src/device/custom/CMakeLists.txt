include(${CMAKE_SOURCE_DIR}/../cmake/customKernelTargets.cmake)

math(EXPR test_size "2 * ${DEVICE_BUFFER_SIZE} * ${NUM_REPLICATIONS} * ${GLOBAL_MEM_UNROLL} * ${VECTOR_COUNT}")

foreach (target ${custom_kernel_targets})
    if (INTELFPGAOPENCL_FOUND)
        add_test(NAME test_${target}_emulation_intel COMMAND STREAM_FPGA_intel -f ${target}_emulate.aocx -n 1 -s ${test_size}
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    endif()

    if (Vitis_FOUND)
        add_test(NAME test_${target}_emulation_xilinx COMMAND STREAM_FPGA_xilinx -f ${target}_emulate.xclbin -n 1 -s ${test_size}
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    endif()
endforeach(target)