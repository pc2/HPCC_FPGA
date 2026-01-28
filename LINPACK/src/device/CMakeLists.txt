
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(hpl_torus_IEC hpl_torus_PCIE)
    add_test(NAME test_emulation_intel COMMAND ./Linpack_intel -f hpl_torus_PCIE_emulate.aocx -m 2 -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
    add_test(NAME test_output_parsing_intel COMMAND ${TEST_SCRIPTS_DIRECTORY}/evaluation/execute_and_parse.sh ./Linpack_intel ${TEST_HOST_FLAGS} -f hpl_torus_PCIE_emulate.aocx -m 2 -n 1 
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(hpl_torus_PCIE hpl_torus_ACCL_buffers)
    add_test(NAME test_emulation_xilinx COMMAND ./Linpack_xilinx -f hpl_torus_PCIE_emulate.xclbin -m 2 -n 1 ${TEST_HOST_FLAGS}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
    add_test(NAME test_output_parsing_xilinx COMMAND ${TEST_SCRIPTS_DIRECTORY}/evaluation/execute_and_parse.sh ./Linpack_xilinx ${TEST_HOST_FLAGS} -f hpl_torus_PCIE_emulate.xclbin -m 2 -n 1 
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})
endif()

