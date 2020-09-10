set(KERNEL_REPLICATION_ENABLED Yes CACHE INTERNAL "Enables kernel replication in the CMake target genertion function")

include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)


if (INTELFPGAOPENCL_FOUND)
generate_kernel_targets_intel(random_access_kernels_single)
add_test(NAME test_emulation_intel COMMAND RandomAccess_intel -f random_access_kernels_single_emulate.aocx -d 1048576 -n 1
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./RandomAccess_intel -f random_access_kernels_single_emulate.aocx -d 1048576 -n 1 
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
if (USE_MPI)
        add_test(NAME test_emulation_mpi_intel COMMAND mpirun -n 2 ./RandomAccess_intel -f random_access_kernels_single_emulate.aocx -d 1048576 -n 1
                    WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
endif()

if (VITIS_FOUND)
        generate_kernel_targets_xilinx(random_access_kernels_single)
        add_test(NAME test_emulation_xilinx COMMAND RandomAccess_xilinx -f random_access_kernels_single_emulate.xclbin -d 1048576 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./RandomAccess_xilinx -f random_access_kernels_single_emulate.xclbin -d 1048576 -n 1 
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        if (USE_MPI)
                add_test(NAME test_emulation_mpi_xilinx COMMAND mpirun -n 2 ./RandomAccess_xilinx -f random_access_kernels_single_emulate.xclbin -d 1048576 -n 1
                            WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        endif()
endif()
