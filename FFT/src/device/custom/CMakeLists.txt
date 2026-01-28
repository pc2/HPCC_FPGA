include(${CMAKE_SOURCE_DIR}/../cmake/customKernelTargets.cmake)

foreach (target ${custom_kernel_targets})
    if (INTELFPGAOPENCL_FOUND)
        add_test(NAME test_${target}_emulation_intel COMMAND FFT_intel -f ${target}_emulate.aocx -n 1 -i 1 WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
    endif()

    if (Vitis_FOUND)
        add_test(NAME test_${target}_emulation_xilinx COMMAND FFT_xilinx -f ${target}_emulate.xclbin -n 1 -i 1 WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
    endif()
endforeach(target)