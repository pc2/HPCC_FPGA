
set(COMPILER_INCLUDES "-I${CMAKE_BINARY_DIR}/src/common/" "-I${CMAKE_CURRENT_SOURCE_DIR}")

set(Vitis_EMULATION_CONFIG_UTIL $ENV{XILINX_VITIS}/bin/emconfigutil)

if (CMAKE_BUILD_TYPE EQUAL "Debug")
        set(VPP_FLAGS "-O0 -g")
else()
        set(VPP_FLAGS "-O3")
endif()

set(file_endings "cl" "cpp" )

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
        string(REGEX MATCH ".*_ACCL.*" is_accl_kernel ${kernel_file_name})
        if (is_accl_kernel AND NOT USE_ACCL)
            continue()
        endif()
        set(file_exists No)
        if (DEFINED FORCE_FILE_ENDING)
                set(file_endings ${FORCE_FILE_ENDING})
        endif()
        foreach (ending ${file_endings})
            set(search_file_name "${CMAKE_SOURCE_DIR}/${base_file_part}.${ending}")
            if (NOT file_exists AND EXISTS ${search_file_name})
                set(file_exists Yes)
                set(selected_file_ending ${ending})
                set(base_file "${search_file_name}")
            endif()
        endforeach()
        if (KERNEL_REPLICATION_ENABLED)
            set(source_f "${CMAKE_BINARY_DIR}/${base_file_part}_replicated_xilinx.${selected_file_ending}")
        else()
            set(source_f "${CMAKE_BINARY_DIR}/${base_file_part}_copied_xilinx.${selected_file_ending}")
        endif()
        if (DEFINED XILINX_KERNEL_NAMES)
            set(bitstream_compile "")
            set(bitstream_compile_emulate "")
            foreach (kernel ${XILINX_KERNEL_NAMES})
                list(APPEND bitstream_compile xilinx_tmp_compile/${kernel_file_name}/${kernel}.xo)
                list(APPEND bitstream_compile_emulate xilinx_tmp_compile/${kernel_file_name}/${kernel}_emulate.xo)
            endforeach()
        else()
            set(bitstream_compile xilinx_tmp_compile/${kernel_file_name}.xo)
            set(bitstream_compile_emulate xilinx_tmp_compile/${kernel_file_name}_emulate.xo)
        endif()
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
        if (USE_ACCL AND is_accl_kernel)
            list(APPEND additional_xos ${ACCL_XOS})
        endif()
        if (USE_UDP)
                list(APPEND additional_xos ${UDP_XOS})
                list(APPEND local_harware_only_flags ${UDP_LINK_CONFIG})
        endif()
        set(xilinx_report_folder "${EXECUTABLE_OUTPUT_PATH}/xilinx_reports")
        set(local_CLFLAGS -DXILINX_FPGA)
        list(APPEND local_CLFLAGS --report_dir=${xilinx_report_folder} --log_dir=${xilinx_report_folder}/logs)
        if (is_accl_kernel)
            list(APPEND local_harware_only_flags ${ACCL_LINK_CONFIG})
        endif()
        string(REGEX MATCH "^.+\.tcl" is_tcl_script ${XILINX_COMPILE_SETTINGS_FILE})
        if (is_tcl_script)
                set(CLFLAGS --hls.pre_tcl ${XILINX_COMPILE_SETTINGS_FILE})
        else()
                set(CLFLAGS --config ${XILINX_COMPILE_SETTINGS_FILE})
        endif()
        list(APPEND local_CLFLAGS ${CLFLAGS})

        # build emulation config for device
        add_custom_command(OUTPUT ${EXECUTABLE_OUTPUT_PATH}/emconfig.json
        COMMAND ${Vitis_EMULATION_CONFIG_UTIL} -f ${FPGA_BOARD_NAME} --od ${EXECUTABLE_OUTPUT_PATH}
        )
        if (XILINX_GENERATE_LINK_SETTINGS)
            add_custom_command(OUTPUT ${xilinx_link_settings}
                    COMMAND ${Python3_EXECUTABLE} ${CODE_GENERATOR} -o ${xilinx_link_settings} -p num_replications=${NUM_REPLICATIONS} ${gen_xilinx_link_settings}
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
                        COMMAND ${Python3_EXECUTABLE} ${CODE_GENERATOR} -o ${source_f} -p num_replications=1 -p num_total_replications=${NUM_REPLICATIONS} ${base_file}
                        MAIN_DEPENDENCY ${base_file}
                )
        else()
                add_custom_command(OUTPUT ${source_f}
                        COMMAND cp ${base_file} ${source_f}
                        MAIN_DEPENDENCY ${base_file}
                )
        endif()

        foreach (kernel ${bitstream_compile_emulate})
            if (DEFINED XILINX_KERNEL_NAMES)
                string(REGEX MATCH ".+/(.+)_emulate\.xo" kernel_name ${kernel})
                set(kernel_name_flag -k ${CMAKE_MATCH_1})
            endif()
            add_custom_command(OUTPUT ${kernel}
                    COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} ${VPP_FLAGS} -DKERNEL_${CMAKE_MATCH_1} -DEMULATE -t sw_emu ${kernel_name_flag} ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_COMPILE_FLAGS} -f ${FPGA_BOARD_NAME} -c ${XILINX_COMPILE_FLAGS} -o ${kernel} ${source_f}
                    MAIN_DEPENDENCY ${source_f}
                    DEPENDS ${XILINX_COMPILE_SETTINGS_FILE}
                    )
        endforeach()
        add_custom_command(OUTPUT ${bitstream_emulate_f}
            COMMAND ${Vitis_COMPILER} ${local_CL_FLAGS} ${VPP_FLAGS} -DEMULATE -t sw_emu ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_LINK_FLAGS} -f ${FPGA_BOARD_NAME} -l --config ${xilinx_link_settings} ${XILINX_COMPILE_FLAGS} -o ${bitstream_emulate_f} ${bitstream_compile_emulate}
                DEPENDS ${bitstream_compile_emulate}
                DEPENDS ${xilinx_link_settings}
                )
        foreach (kernel ${bitstream_compile})
            if (DEFINED XILINX_KERNEL_NAMES)
                string(REGEX MATCH ".+/(.+)\.xo" kernel_name ${kernel})
                set(kernel_name_flag -k ${CMAKE_MATCH_1})
            endif()
            add_custom_command(OUTPUT ${kernel}
                    COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} ${VPP_FLAGS} -t hw -DKERNEL_${CMAKE_MATCH_1} ${kernel_name_flag} ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_COMPILE_FLAGS}  --platform ${FPGA_BOARD_NAME} -R2 -c ${XILINX_COMPILE_FLAGS} -o ${kernel} ${source_f}
                    MAIN_DEPENDENCY ${source_f}
                    DEPENDS ${XILINX_COMPILE_SETTINGS_FILE}
                    )
        endforeach()
        add_custom_command(OUTPUT ${bitstream_f}
                COMMAND ${Vitis_COMPILER} ${local_CLFLAGS} ${VPP_FLAGS} ${local_harware_only_flags} -t hw ${COMPILER_INCLUDES} ${XILINX_ADDITIONAL_LINK_FLAGS} --platform ${FPGA_BOARD_NAME} -R2 -l --config ${xilinx_link_settings} -o ${bitstream_f} ${additional_xos} ${bitstream_compile}
                DEPENDS ${bitstream_compile}
                DEPENDS ${xilinx_link_settings}
                )
        add_custom_target(${kernel_file_name}_emulate_xilinx
		DEPENDS ${bitstream_emulate_f}
                DEPENDS ${source_f} ${CMAKE_BINARY_DIR}/src/common/parameters.h ${EXECUTABLE_OUTPUT_PATH}/emconfig.json)
        add_custom_target(${kernel_file_name}_xilinx
		DEPENDS ${bitstream_f}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_target(${kernel_file_name}_report_xilinx
		DEPENDS ${bitstream_compile}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        if(USE_ACCL AND is_accl_kernel)
            add_dependencies(${kernel_file_name}_xilinx accl_device)
        endif()
        if (USE_UDP)
                add_dependencies(${kernel_file_name}_xilinx vnx_udp_stack)
        endif()
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
        set(report_f ${kernel_file_name}_report_intel/reports/report.html)
        set(bitstream_emulate_f ${kernel_file_name}_emulate.aocx)
        set(bitstream_f ${kernel_file_name}.aocx)
        if (KERNEL_REPLICATION_ENABLED)
                set(codegen_parameters -p num_replications=${NUM_REPLICATIONS} -p num_total_replications=${NUM_REPLICATIONS})
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
        add_custom_command(OUTPUT ${EXECUTABLE_OUTPUT_PATH}/${bitstream_emulate_f}
                COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_emulate_f} ${EXECUTABLE_OUTPUT_PATH}/${bitstream_emulate_f}
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_emulate_f}
        )
        add_custom_command(OUTPUT ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_reports/report.html
                COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_BINARY_DIR}/${kernel_file_name}_report_intel/reports/ ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_reports/
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${report_f}
        )
        add_custom_command(OUTPUT ${EXECUTABLE_OUTPUT_PATH}/${bitstream_f}
                COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_f} ${EXECUTABLE_OUTPUT_PATH}/${bitstream_f}
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/${kernel_file_name}/reports ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_synth_reports
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${kernel_file_name}/acl_quartus_report.txt ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_synth_reports/acl_quartus_report.txt
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${kernel_file_name}/quartus_sh_compile.log ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_synth_reports/quartus_sh_compile.log
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${kernel_file_name}/${kernel_file_name}.log ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_synth_reports/${kernel_file_name}.log
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_f}
        )
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_emulate_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} -DEMULATE -DINTEL_FPGA ${COMPILER_INCLUDES} ${AOC_FLAGS} -march=emulator
                -o ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_emulate_f}
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} -DINTEL_FPGA ${COMPILER_INCLUDES} ${AOC_FLAGS} -board=${FPGA_BOARD_NAME}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${bitstream_f}
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${report_f}
                COMMAND ${IntelFPGAOpenCL_AOC} ${source_f} -DINTEL_FPGA ${COMPILER_INCLUDES} ${AOC_FLAGS} -rtl -report -board=${FPGA_BOARD_NAME}
                -o ${kernel_file_name}_report_intel
                MAIN_DEPENDENCY ${source_f}
                DEPENDS ${CMAKE_BINARY_DIR}/src/common/parameters.h
                )
        add_custom_target(${kernel_file_name}_report_intel
                DEPENDS ${EXECUTABLE_OUTPUT_PATH}/${kernel_file_name}_reports/report.html)
        add_custom_target(${kernel_file_name}_intel
                DEPENDS ${EXECUTABLE_OUTPUT_PATH}/${bitstream_f})
        add_custom_target(${kernel_file_name}_emulate_intel
                DEPENDS ${EXECUTABLE_OUTPUT_PATH}/${bitstream_emulate_f})
        list(APPEND kernel_emulation_targets_intel ${kernel_file_name}_emulate_intel)
        set(kernel_emulation_targets_intel ${kernel_emulation_targets_intel} CACHE INTERNAL "Kernel emulation targets used to define dependencies for the tests for intel devices")
    endforeach ()
endfunction()
