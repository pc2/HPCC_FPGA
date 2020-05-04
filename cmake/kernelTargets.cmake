
set(COMPILER_INCLUDES "-I${CMAKE_CURRENT_BINARY_DIR}/../common")
set(CLFLAGS --config ${XILINX_COMPILE_SETTINGS_FILE})

set(Vitis_EMULATION_CONFIG_UTIL $ENV{XILINX_VITIS}/bin/emconfigutil)

##
# This function will create build targets for the kernels for emulationand synthesis for xilinx.
##
function(generate_kernel_targets_xilinx)
    foreach (kernel_file_name ${ARGN})
        set(base_file "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl")
        if (KERNEL_REPLICATION_ENABLED)
            set(source_f "${CMAKE_BINARY_DIR}/src/device/replicated_${kernel_file_name}_xilinx.cl")
        else()
            set(source_f ${base_file})
        endif()
        set(bitstream_compile ${EXECUTABLE_OUTPUT_PATH}/xilinx_tmp_compile/${kernel_file_name}.xo)
        set(bitstream_compile_emulate ${EXECUTABLE_OUTPUT_PATH}/xilinx_tmp_compile/${kernel_file_name}_emulate.xo)
        set(bitstream_emulate_f
            ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_emulate.xclbin)
        set(bitstream_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}.xclbin)
        if (XILINX_GENERATE_LINK_SETTINGS)
		set(gen_xilinx_link_settings ${XILINX_LINK_SETTINGS_FILE})
            set(xilinx_link_settings ${CMAKE_BINARY_DIR}/settings/settings.link.xilinx.${kernel_file_name}.ini)
        else()
            set(gen_xilinx_link_settings ${XILINX_LINK_SETTINGS_FILE})
            set(xilinx_link_settings ${XILINX_LINK_SETTINGS_FILE})
        endif()
        set(xilinx_report_folder "--report_dir=${EXECUTABLE_OUTPUT_PATH}/xilinx_reports")
        file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_reports)
        set(local_CLFLAGS ${CLFLAGS})
        list(APPEND local_CLFLAGS ${xilinx_report_folder} --log_dir=${EXECUTABLE_OUTPUT_PATH}/xilinx_tmp_compile)

        # build emulation config for device
        add_custom_command(OUTPUT ${EXECUTABLE_OUTPUT_PATH}/emconfig.json
        COMMAND ${Vitis_EMULATION_CONFIG_UTIL} -f ${FPGA_BOARD_NAME} --od ${EXECUTABLE_OUTPUT_PATH}
        )
        if (XILINX_GENERATE_LINK_SETTINGS)
            add_custom_command(OUTPUT ${xilinx_link_settings}
                    COMMAND ${CMAKE_COMMAND} -Dsettings_f=${xilinx_link_settings} -Dbase_file=${gen_xilinx_link_settings} -DNUM_REPLICATIONS=${NUM_REPLICATIONS} -P "${CMAKE_SOURCE_DIR}/../cmake/generateXilinxSettings.cmake"
                    MAIN_DEPENDENCY ${gen_xilinx_link_settings}
                    )
        endif()

        add_custom_command(OUTPUT ${source_f}
                COMMAND ${CMAKE_COMMAND} -Dsource_f=${source_f} -Dbase_file=${base_file} -DNUM_REPLICATIONS=1 -P "${CMAKE_SOURCE_DIR}/../cmake/generateKernels.cmake"
                MAIN_DEPENDENCY ${base_file}
                )

        add_custom_command(OUTPUT ${bitstream_compile_emulate}
                COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} -t sw_emu ${COMPILER_INCLUDES} -f ${FPGA_BOARD_NAME} -g -c ${XILINX_COMPILE_FLAGS} -o ${bitstream_compile_emulate} ${source_f}
                MAIN_DEPENDENCY ${source_f}
                )
        add_custom_command(OUTPUT ${bitstream_emulate_f}
            COMMAND ${Vitis_COMPILER} ${local_CL_FLAGS} -t sw_emu ${COMPILER_INCLUDES} -f ${FPGA_BOARD_NAME} -g -l --config ${xilinx_link_settings} ${XILINX_COMPILE_FLAGS} -o ${bitstream_emulate_f} ${bitstream_compile_emulate}
                MAIN_DEPENDENCY ${bitstream_compile_emulate}
                DEPENDS ${xilinx_link_settings}
                )
        add_custom_command(OUTPUT ${bitstream_compile}
                COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} -t hw ${COMPILER_INCLUDES} --platform ${FPGA_BOARD_NAME} -R2 -c ${XILINX_COMPILE_FLAGS} -o ${bitstream_compile} ${source_f}
                MAIN_DEPENDENCY ${source_f}
                )
        add_custom_command(OUTPUT ${bitstream_f}
                COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} -t hw ${COMPILER_INCLUDES} --platform ${FPGA_BOARD_NAME} -R2 -l --config ${xilinx_link_settings} ${XILINX_COMPILE_FLAGS} -o ${bitstream_f} ${bitstream_compile}
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
        set(base_file "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl")
        if (KERNEL_REPLICATION_ENABLED)
                set(source_f "${CMAKE_BINARY_DIR}/src/device/replicated_${kernel_file_name}.cl")
        else()
                set(source_f ${base_file})
        endif()
        set(report_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_report_intel)
        set(bitstream_emulate_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_emulate.aocx)
        set(bitstream_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}.aocx)
        add_custom_command(OUTPUT ${source_f}
                COMMAND ${CMAKE_COMMAND} -Dsource_f=${source_f} -Dbase_file=${base_file} -DNUM_REPLICATIONS=${NUM_REPLICATIONS} -P "${CMAKE_SOURCE_DIR}/../cmake/generateKernels.cmake"
                MAIN_DEPENDENCY ${base_file}
                )
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
                DEPENDS ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h
                SOURCES ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h)
        add_custom_target(${kernel_file_name}_intel DEPENDS ${bitstream_f}
                DEPENDS ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h
                SOURCES ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h)
        add_custom_target(${kernel_file_name}_emulate_intel DEPENDS ${bitstream_emulate_f}
                DEPENDS ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h
                SOURCES ${source_f} "${CMAKE_SOURCE_DIR}/src/device/${kernel_file_name}.cl" ${CMAKE_BINARY_DIR}/src/common/parameters.h)
    endforeach ()
endfunction()
