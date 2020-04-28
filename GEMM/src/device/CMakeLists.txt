
set(AOC_INCLUDES "-I${CMAKE_CURRENT_BINARY_DIR}/../common")

function(generate_kernel_targets)
    foreach (kernel_file_name ${ARGN})
        set(source_f ${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file_name}.cl)
        set(report_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_report)
        set(bitstream_emulate_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_emulate.aocx)
        set(bitstream_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}.aocx)
        set(out_f "${CMAKE_CURRENT_BINARY_DIR}/${out_f}")
        add_custom_command(OUTPUT ${bitstream_emulate_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} ${AOC_INCLUDES} ${AOC_FLAGS} -legacy-emulator -march=emulator
                -o ${bitstream_emulate_f}
                )
        add_custom_command(OUTPUT ${bitstream_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} ${AOC_INCLUDES} ${AOC_FLAGS} -board=${FPGA_BOARD_NAME}
                -o ${bitstream_f}
                )
        add_custom_command(OUTPUT ${report_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} ${AOC_INCLUDES} ${AOC_FLAGS} -rtl -report -board=${FPGA_BOARD_NAME}
                -o ${report_f}
                )
        add_custom_target(${kernel_file_name}_report DEPENDS ${report_f}
                DEPENDS ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h
                SOURCES ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h)
        add_custom_target(${kernel_file_name} DEPENDS ${bitstream_f}
                DEPENDS ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h
                SOURCES ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h)
        add_custom_target(${kernel_file_name}_emulate DEPENDS ${bitstream_emulate_f}
                DEPENDS ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h
                SOURCES ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h)
    endforeach ()
endfunction()

generate_kernel_targets(gemm_cannon)
add_test(NAME test_emulation_intel COMMAND ./fgemm -f gemm_cannon_emulate.aocx -m ${BLOCK_SIZE}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
