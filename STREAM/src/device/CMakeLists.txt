
set(KERNEL_REPLICATION_ENABLED Yes CACHE INTERNAL "Enables kernel replication in the CMake target genertion function")

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

math(EXPR test_size "2 * ${DEVICE_BUFFER_SIZE} * ${NUM_REPLICATIONS} * ${GLOBAL_MEM_UNROLL} * ${VECTOR_COUNT}")

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(stream_kernels_single)
    add_test(NAME test_single_emulation_intel COMMAND STREAM_FPGA_intel -f stream_kernels_single_emulate.aocx -n 1 -s ${test_size}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./STREAM_FPGA_intel -s ${test_size} -f stream_kernels_single_emulate.aocx -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    if (USE_MPI)
            add_test(NAME test_emulation_mpi_intel COMMAND mpirun -n 2 ./STREAM_FPGA_intel -f stream_kernels_single_emulate.aocx -n 1 -s ${test_size}
                        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    endif()
endif()

if (VITIS_FOUND)
    generate_kernel_targets_xilinx(stream_kernels_single)
    add_test(NAME test_single_emulation_xilinx COMMAND STREAM_FPGA_xilinx -f stream_kernels_single_emulate.xclbin -n 1 -s ${test_size}
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./STREAM_FPGA_xilinx -s ${test_size} -f stream_kernels_single_emulate.xclbin -n 1 
            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    if (USE_MPI)
            add_test(NAME test_emulation_mpi_xilinx COMMAND mpirun -n 2 ./STREAM_FPGA_xilinx -f stream_kernels_single_emulate.xclbin -n 1 -s ${test_size}
                        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    endif()
endif()
