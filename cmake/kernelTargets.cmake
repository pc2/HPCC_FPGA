
set(COMPILER_INCLUDES "-I${CMAKE_BINARY_DIR}/src/common/" "-I${CMAKE_CURRENT_SOURCE_DIR}")
set(CLFLAGS --config ${XILINX_COMPILE_SETTINGS_FILE})

set(Vitis_EMULATION_CONFIG_UTIL $ENV{XILINX_VITIS}/bin/emconfigutil)

##
# This function will create build targets for the kernels for emulationand synthesis for xilinx.
##
function(generate_kernel_targets_xilinx)
    foreach (kernel_file_name ${ARGN})
        string(REGEX MATCH "^custom_.*" is_custom_kernel ${kernel_file_name})
        if (is_custom_kernel) 
                string(REPLACE "custom_" "" base_file_name ${kernel_file_name})
                set(base_file_part "src/device/custom/${base_file_name}")
        else()
                set(base_file_part "src/device/${kernel_file_name}")
        endif()
        set(base_file "${CMAKE_SOURCE_DIR}/${base_file_part}.cl")
        if (KERNEL_REPLICATION_ENABLED)
            set(source_f "${CMAKE_BINARY_DIR}/${base_file_part}_replicated_xilinx.cl")
        else()
            set(source_f "${CMAKE_BINARY_DIR}/${base_file_part}_copied_xilinx.cl")
        endif()
        set(bitstream_compile ${EXECUTABLE_OUTPUT_PATH}/xilinx_tmp_compile/${kernel_file_name}.xo)
        set(bitstream_compile_emulate ${EXECUTABLE_OUTPUT_PATH}/xilinx_tmp_compile/${kernel_file_name}_emulate.xo)
        set(bitstream_emulate_f
            ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_emulate.xclbin)
        set(bitstream_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}.xclbin)
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/settings)
        if (XILINX_GENERATE_LINK_SETTINGS)
	    set(gen_xilinx_link_settings ${XILINX_LINK_SETTINGS_FILE})
            set(xilinx_link_settings ${CMAKE_BINARY_DIR}/settings/settings.link.xilinx.${kernel_file_name}.ini)
        else()
            set(gen_xilinx_link_settings ${XILINX_LINK_SETTINGS_FILE})
            set(xilinx_link_settings ${CMAKE_BINARY_DIR}/settings/settings.link.xilinx.${kernel_file_name}.ini)
        endif()
        set(xilinx_report_folder "--report_dir=${EXECUTABLE_OUTPUT_PATH}/xilinx_reports")
        set(local_CLFLAGS ${CLFLAGS} -DXILINX_FPGA)
        list(APPEND local_CLFLAGS ${xilinx_report_folder} --log_dir=${EXECUTABLE_OUTPUT_PATH}/xilinx_tmp_compile)

        # build emulation config for device
        add_custom_command(OUTPUT ${EXECUTABLE_OUTPUT_PATH}/emconfig.json
        COMMAND ${Vitis_EMULATION_CONFIG_UTIL} -f ${FPGA_BOARD_NAME} --od ${EXECUTABLE_OUTPUT_PATH}
        )
        if (XILINX_GENERATE_LINK_SETTINGS)
            add_custom_command(OUTPUT ${xilinx_link_settings}
                    COMMAND ${Python3_EXECUTABLE} ${CODE_GENERATOR} -o ${xilinx_link_settings} -p num_replications=${NUM_REPLICATIONS} --comment "\"#\"" --comment-ml-start "\"$$\"" --comment-ml-end "\"$$\"" ${gen_xilinx_link_settings}
                    MAIN_DEPENDENCY ${gen_xilinx_link_settings}
                    )
        else()
                add_custom_command(OUTPUT ${xilinx_link_settings}
                        COMMAND cp ${gen_xilinx_link_settings} ${xilinx_link_settings}
                        MAIN_DEPENDENCY ${gen_xilinx_link_settings}
                        )
        endif()

        if (KERNEL_REPLICATION_ENABLED)
                add_custom_command(OUTPUT ${source_f}
                        COMMAND ${Python3_EXECUTABLE} ${CODE_GENERATOR} -o ${source_f} -p num_replications=1 ${base_file}
                        MAIN_DEPENDENCY ${base_file}
                )
        else()
                add_custom_command(OUTPUT ${source_f}
                        COMMAND cp ${base_file} ${source_f}
                        MAIN_DEPENDENCY ${base_file}
                )
        endif()

        add_custom_command(OUTPUT ${bitstream_compile_emulate}
                COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} -t sw_emu ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_COMPILE_FLAGS} -f ${FPGA_BOARD_NAME} -g -c ${XILINX_COMPILE_FLAGS} -o ${bitstream_compile_emulate} ${source_f}
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${XILINX_COMPILE_SETTINGS_FILE}
                )
        add_custom_command(OUTPUT ${bitstream_emulate_f}
            COMMAND ${Vitis_COMPILER} ${local_CL_FLAGS} -t sw_emu ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_LINK_FLAGS} -f ${FPGA_BOARD_NAME} -g -l --config ${xilinx_link_settings} ${XILINX_COMPILE_FLAGS} -o ${bitstream_emulate_f} ${bitstream_compile_emulate}
                MAIN_DEPENDENCY ${bitstream_compile_emulate}
                DEPENDS ${xilinx_link_settings}
                )
        add_custom_command(OUTPUT ${bitstream_compile}
                COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} -t hw ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_COMPILE_FLAGS}  --platform ${FPGA_BOARD_NAME} -R2 -c ${XILINX_COMPILE_FLAGS} -o ${bitstream_compile} ${source_f}
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${XILINX_COMPILE_SETTINGS_FILE}
                )
        add_custom_command(OUTPUT ${bitstream_f}
                COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} -t hw ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_LINK_FLAGS} --platform ${FPGA_BOARD_NAME} -R2 -l --config ${xilinx_link_settings} ${XILINX_COMPILE_FLAGS} -o ${bitstream_f} ${bitstream_compile}
                MAIN_DEPENDENCY ${bitstream_compile}
                DEPENDS ${xilinx_link_settings}
                )
        add_custom_target(${kernel_file_name}_emulate_xilinx 
                MAIN_DEPENDENCY ${bitstream_emulate_f} 
                DEPENDS ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h ${EXECUTABLE_OUTPUT_PATH}/emconfig.json)
        add_custom_target(${kernel_file_name}_xilinx
                MAIN_DEPENDENCY ${bitstream_f} 
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_target(${kernel_file_name}_report_xilinx
                MAIN_DEPENDENCY ${bitstream_compile} 
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        list(APPEND kernel_emulation_targets_xilinx ${kernel_file_name}_emulate_xilinx)
        set(kernel_emulation_targets_xilinx ${kernel_emulation_targets_xilinx} CACHE INTERNAL "Kernel emulation targets used to define dependencies for the tests for Xilinx devices")
    endforeach ()
endfunction()


##
# This function will create build targets for the kernels for emulation, reports and synthesis.
# It will use the generate_kernel_replication function to generate a new code file containing the code for all kernels
##
function(generate_kernel_targets_intel)
    foreach (kernel_file_name ${ARGN})
        string(REGEX MATCH "^custom_.*" is_custom_kernel ${kernel_file_name})
        if (is_custom_kernel) 
                string(REPLACE "custom_" "" base_file_name ${kernel_file_name})
                set(base_file_part "src/device/custom/${base_file_name}")
        else()
                set(base_file_part "src/device/${kernel_file_name}")
        endif()
        set(base_file "${CMAKE_SOURCE_DIR}/${base_file_part}.cl")
        if (KERNEL_REPLICATION_ENABLED)
            set(source_f "${CMAKE_BINARY_DIR}/${base_file_part}_replicated_intel.cl")
        else()
            set(source_f "${CMAKE_BINARY_DIR}/${base_file_part}_copied_intel.cl")
        endif()
        set(report_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_report_intel)
        set(bitstream_emulate_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_emulate.aocx)
        set(bitstream_f ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}.aocx)
        if (KERNEL_REPLICATION_ENABLED)
                set(codegen_parameters -p num_replications=${NUM_REPLICATIONS})
                if (INTEL_CODE_GENERATION_SETTINGS)
                        list(APPEND codegen_parameters -p "\"use_file('${INTEL_CODE_GENERATION_SETTINGS}')\"")
                endif()
                add_custom_command(OUTPUT ${source_f}
                        COMMAND ${Python3_EXECUTABLE} ${CODE_GENERATOR} -o ${source_f} ${codegen_parameters} ${base_file}
                        MAIN_DEPENDENCY ${base_file}
                        )
        else()
                add_custom_command(OUTPUT ${source_f}
                        COMMAND cp ${base_file} ${source_f}
                        MAIN_DEPENDENCY ${base_file}
                )
        endif()
        add_custom_command(OUTPUT ${bitstream_emulate_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} -DINTEL_FPGA ${COMPILER_INCLUDES} ${AOC_FLAGS} -legacy-emulator -march=emulator
                -o ${bitstream_emulate_f}
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_command(OUTPUT ${bitstream_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} -DINTEL_FPGA ${COMPILER_INCLUDES} ${AOC_FLAGS} -board=${FPGA_BOARD_NAME}
                -o ${bitstream_f}
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_command(OUTPUT ${report_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} -DINTEL_FPGA ${COMPILER_INCLUDES} ${AOC_FLAGS} -rtl -report -board=${FPGA_BOARD_NAME}
                -o ${report_f}
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_target(${kernel_file_name}_report_intel 
                DEPENDS ${report_f})
        add_custom_target(${kernel_file_name}_intel 
                DEPENDS ${bitstream_f})
        add_custom_target(${kernel_file_name}_emulate_intel 
                DEPENDS ${bitstream_emulate_f})
        list(APPEND kernel_emulation_targets_intel ${kernel_file_name}_emulate_intel)
        set(kernel_emulation_targets_intel ${kernel_emulation_targets_intel} CACHE INTERNAL "Kernel emulation targets used to define dependencies for the tests for intel devices")
    endforeach ()
endfunction()
