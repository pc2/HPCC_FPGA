
set(COMPILER_INCLUDES "-I${CMAKE_CURRENT_BINARY_DIR}/../common")
set(CLFLAGS --config ${XILINX_COMPILE_SETTINGS})

set(Vitis_EMULATION_CONFIG_UTIL $ENV{XILINX_VITIS}/bin/emconfigutil)

##
# This function will create build targets for the kernels for emulationand synthesis for xilinx.
##
function(generate_kernel_targets_xilinx)
    foreach (kernel_file_name ${ARGN})
        set(base_file "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl")
        set(source_f "${CMAKE_BINARY_DIR}/src/device/replicated_${kernel_file_name}_xilinx.cl")
        set(bitstream_compile ${EXECUTABLE_OUTPUT_PATH}/tmp_compile/${kernel_file_name}.xo)
        set(bitstream_compile_emulate ${EXECUTABLE_OUTPUT_PATH}/tmp_compile/${kernel_file_name}_emulate.xo)
        set(bitstream_emulate_f
            ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_emulate.xclbin)
        set(bitstream_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}.xclbin)
        set(xilinx_link_settings ${XILINX_LINK_SETTINGS_FILE})

        # build emulation config for device
        add_custom_command(OUTPUT ${EXECUTABLE_OUTPUT_PATH}/emconfig.json
        COMMAND ${Vitis_EMULATION_CONFIG_UTIL} -f ${FPGA_BOARD_NAME} --od ${EXECUTABLE_OUTPUT_PATH}
        )

        add_custom_command(OUTPUT ${source_f}
                COMMAND ${CMAKE_COMMAND} -Dsource_f=${source_f} -Dbase_file=${base_file} -DNUM_REPLICATIONS=1 -P "${CMAKE_SOURCE_DIR}/src/device/generateKernels.cmake"
                MAIN_DEPENDENCY ${base_file}
                )

        add_custom_command(OUTPUT ${bitstream_compile_emulate}
                COMMAND ${Vitis_COMPILER} ${CLFLAGS} -t sw_emu ${COMPILER_INCLUDES} -f ${FPGA_BOARD_NAME} -g -c ${XILINX_COMPILE_FLAGS} -o ${bitstream_compile_emulate} ${source_f}
                MAIN_DEPENDENCY ${source_f}
                )
        add_custom_command(OUTPUT ${bitstream_emulate_f}
            COMMAND ${Vitis_COMPILER} -t sw_emu ${COMPILER_INCLUDES} -f ${FPGA_BOARD_NAME} -g -l --config ${xilinx_link_settings} ${XILINX_COMPILE_FLAGS} -o ${bitstream_emulate_f} ${bitstream_compile_emulate}
                MAIN_DEPENDENCY ${bitstream_compile_emulate}
                DEPENDS ${xilinx_link_settings}
                )
        add_custom_command(OUTPUT ${bitstream_compile}
                COMMAND ${Vitis_COMPILER} ${CLFLAGS} -t hw ${COMPILER_INCLUDES} --platform ${FPGA_BOARD_NAME} -R2 -c ${XILINX_COMPILE_FLAGS} -o ${bitstream_compile} ${source_f}
                MAIN_DEPENDENCY ${source_f}
                )
        add_custom_command(OUTPUT ${bitstream_f}
                COMMAND ${Vitis_COMPILER} ${CLFLAGS} -t hw ${COMPILER_INCLUDES} --platform ${FPGA_BOARD_NAME} -R2 -l --config ${xilinx_link_settings} ${XILINX_COMPILE_FLAGS} -o ${bitstream_f} ${bitstream_compile}
                MAIN_DEPENDENCY ${bitstream_compile}
                DEPENDS ${xilinx_link_settings}
                )
        add_custom_target(${kernel_file_name}_emulate_xilinx DEPENDS ${bitstream_emulate_f} ${EXECUTABLE_OUTPUT_PATH}/emconfig.json
                DEPENDS ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h
                SOURCES ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h)
        add_custom_target(${kernel_file_name}_xilinx
                DEPENDS ${bitstream_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_target(${kernel_file_name}_compile_xilinx
                DEPENDS ${bitstream_compile} ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
    endforeach ()
endfunction()


##
# This function will create build targets for the kernels for emulation, reports and synthesis.
# It will use the generate_kernel_replication function to generate a new code file containing the code for all kernels
##
function(generate_kernel_targets_intel)
    foreach (kernel_file_name ${ARGN})
        set(source_f "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl")
        set(report_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_report_intel)
        set(bitstream_emulate_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_emulate.aocx)
        set(bitstream_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}.aocx)

        add_custom_command(OUTPUT ${bitstream_emulate_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} ${COMPILER_INCLUDES} ${AOC_FLAGS} -legacy-emulator -march=emulator
                -o ${bitstream_emulate_f}
                MAIN_DEPENDENCY ${source_f}
                )
        add_custom_command(OUTPUT ${bitstream_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} ${COMPILER_INCLUDES} ${AOC_FLAGS} -board=${FPGA_BOARD_NAME}
                -o ${bitstream_f}
                MAIN_DEPENDENCY ${source_f}
                )
        add_custom_command(OUTPUT ${report_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} ${COMPILER_INCLUDES} ${AOC_FLAGS} -rtl -report -board=${FPGA_BOARD_NAME}
                -o ${report_f}
                MAIN_DEPENDENCY ${source_f}
                )
        add_custom_target(${kernel_file_name}_report_intel DEPENDS ${report_f}
                DEPENDS ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h)
        add_custom_target(${kernel_file_name}_intel DEPENDS ${bitstream_f}
                DEPENDS ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h)
        add_custom_target(${kernel_file_name}_emulate_intel DEPENDS ${bitstream_emulate_f}
                DEPENDS ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h)
    endforeach ()
endfunction()

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(lu_blocked_pvt lu_blocked_pvt_test)
    add_test(NAME test_emulation_intel COMMAND STREAM_FPGA_intel -f stream_kernels_emulate.aocx)
endif()

