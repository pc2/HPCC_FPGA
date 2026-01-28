include(${CMAKE_SOURCE_DIR}/../cmake/customKernelTargets.cmake)

foreach (target ${custom_kernel_targets})
    if (INTELFPGAOPENCL_FOUND)
        add_test(NAME test_${target}_emulation_intel COMMAND ./RandomAccess_intel -f ${target}_emulate.aocx -d 1048576 -n 1
                WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
    endif()

    if (Vitis_FOUND)
        add_test(NAME test_${target}_emulation_xilinx COMMAND ./RandomAccess_xilinx -f ${target}_emulate.xclbin -d 1048576 -n 1
                WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
    endif()
endforeach(target)