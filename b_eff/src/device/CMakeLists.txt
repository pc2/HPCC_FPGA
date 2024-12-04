
set(KERNEL_REPLICATION_ENABLED Yes CACHE INTERNAL "Enables kernel replication in the CMake target genertion function")
set(NUM_REPLICATIONS 2)
include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
        generate_kernel_targets_intel(communication_bw520n_IEC communication_PCIE)
        add_test(NAME test_emulation_iec_intel COMMAND ${CMAKE_SOURCE_DIR}/scripts/clean_emulation_output_files.sh ${CMAKE_BINARY_DIR} ./Network_intel -f communication_bw520n_IEC_emulate.aocx -l 1 -u 10 --min-size 6 -m 6 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_emulation_cpu_intel COMMAND ${CMAKE_SOURCE_DIR}/scripts/clean_emulation_output_files.sh ${CMAKE_BINARY_DIR} ./Network_intel -f communication_PCIE_emulate.aocx --comm-type CPU -l 1 -u 10 -m 0 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_emulation_pcie_intel COMMAND ${CMAKE_SOURCE_DIR}/scripts/clean_emulation_output_files.sh ${CMAKE_BINARY_DIR} ./Network_intel -f communication_PCIE_emulate.aocx -l 1 -u 10 -m 0 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_output_parsing_intel COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Network_intel -f communication_PCIE_emulate.aocx -l 1 -u 1 -m 20 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

if (Vitis_FOUND)
        generate_kernel_targets_xilinx(communication_PCIE communication_UDP)
        if (USE_ACCL)
                generate_kernel_targets_xilinx(communication_ACCL communication_ACCL_pl
                    communication_ACCL_pl_stream)
        endif()
        add_test(NAME test_emulation_pcie_xilinx COMMAND ${CMAKE_SOURCE_DIR}/scripts/clean_emulation_output_files.sh ${CMAKE_BINARY_DIR} ./Network_xilinx -f communication_PCIE_emulate.xclbin -l 1 -u 10 -m 0 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_emulation_cpu_xilinx COMMAND ${CMAKE_SOURCE_DIR}/scripts/clean_emulation_output_files.sh ${CMAKE_BINARY_DIR} ./Network_xilinx -f communication_PCIE_emulate.xclbin --comm-type CPU -l 1 -u 10 -m 0 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_output_parsing_xilinx COMMAND ${CMAKE_SOURCE_DIR}/../scripts/evaluation/execute_and_parse.sh ./Network_xilinx -f communication_PCIE_emulate.xclbin -l 1 -u 1 -m 20 -n 1
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_emulation_pcie_reverse_xilinx COMMAND ${CMAKE_SOURCE_DIR}/scripts/clean_emulation_output_files.sh ${CMAKE_BINARY_DIR} ./Network_xilinx -f communication_PCIE_emulate.xclbin -l 1 -u 10 -m 0 -n 1 --kernel-latency --pcie-write --pcie-read
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
        add_test(NAME test_emulation_pcie_reverse_batch_xilinx COMMAND ${CMAKE_SOURCE_DIR}/scripts/clean_emulation_output_files.sh ${CMAKE_BINARY_DIR} ./Network_xilinx -f communication_PCIE_emulate.xclbin -l 1 -u 10 -m 0 -n 1 --kernel-latency --pcie-write --pcie-read --pcie-batch
                WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()
