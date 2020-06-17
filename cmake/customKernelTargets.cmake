# This file will create build targets for custom kernels
# and adds all these kernel to the variable kernel_emulation_targets
# which is used as a dependency of the unit tests
#
# sets custom_kernel_targets to a list of found kernel targets


include(${CMAKE_SOURCE_DIR}/../cmake/kernelTargets.cmake)

file(GLOB custom_kernel_files
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cl"
)

set(custom_kernel_targets "")
foreach(file_name ${custom_kernel_files})
    string(REPLACE ".cl" "" target_name ${file_name})
    string(PREPEND target_name "custom_")
    list(APPEND custom_kernel_targets ${target_name})
endforeach(file_name)
set(custom_kernel_targets ${custom_kernel_targets} CACHE INTERNAL "Custom kernel targets that are used to create Ctests in the benchmark specific CMakeLists")

if (INTELFPGAOPENCL_FOUND)
    generate_kernel_targets_intel(${custom_kernel_targets})
endif()

if (Vitis_FOUND)
    generate_kernel_targets_xilinx(${custom_kernel_targets})
endif()