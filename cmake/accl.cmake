
set(ACCL_STACK_TYPE "UDP" CACHE STRING "Network layer used in ACCL")
set(ACCL_UDP_ETH_IF 0 CACHE STRING "Ethernet interface used. On ETHZ: 0 = switch, 1 = direct")
set(ACCL_DEVICE_NAME "xcu280-fsvh2892-2L-e" CACHE STRING "Name of the FPGA used on the target platform")

set(ACCL_CCLO_KERNEL_DIR ${extern_accl_SOURCE_DIR}/kernels/cclo/)
set(ACCL_CCLO_KERNEL_XO cclo_offload.xo)

set(ACCL_HARDWARE_DIR ${extern_accl_SOURCE_DIR}/test/hardware)
set(ACCL_VNX_DIR ${ACCL_HARDWARE_DIR}/xup_vitis_network_example/)
set(ACCL_UDP_MAC_XO ${ACCL_VNX_DIR}/Ethernet/_x.${FPGA_BOARD_NAME}/cmac_${ACCL_UDP_ETH_IF}.xo)
set(ACCL_UDP_NET_XO ${ACCL_VNX_DIR}/NetLayers/_x.${FPGA_BOARD_NAME}/networklayer.xo)

add_custom_command(
    OUTPUT ${ACCL_CCLO_KERNEL_DIR}/${ACCL_CCLO_KERNEL_XO}
    COMMAND make STACK_TYPE=${ACCL_STACK_TYPE} PLATFORM=${FPGA_BOARD_NAME}
    WORKING_DIRECTORY ${ACCL_CCLO_KERNEL_DIR})

add_custom_command(
    OUTPUT ${ACCL_UDP_MAC_XO}
    COMMAND make -C ${ACCL_VNX_DIR}/Ethernet DEVICE=${FPGA_BOARD_NAME} INTERFACE=${ACCL_UDP_ETH_IF} all
    WORKING_DIRECTORY ${ACCL_HARDWARE_DIR}) 

add_custom_command(
    OUTPUT ${ACCL_UDP_NET_XO}
    COMMAND make -C ${ACCL_VNX_DIR}/NetLayers DEVICE=${FPGA_BOARD_NAME} all
    WORKING_DIRECTORY ${ACCL_HARDWARE_DIR}) 


set(ACCL_PLUGINS_DIR ${extern_accl_SOURCE_DIR}/kernels/plugins)
set(ACCL_PLUGINS_HOSTCTRL ${ACCL_PLUGINS_DIR}/hostctrl/hostctrl.xo)
set(ACCL_PLUGINS_SUM ${ACCL_PLUGINS_DIR}/reduce_sum/reduce_sum.xo)
set(ACCL_PLUGINS_COMPRESSION ${ACCL_PLUGINS_DIR}/hp_compression/hp_compression.xo)
set(ACCL_PLUGINS_LOOPBACK ${ACCL_PLUGINS_DIR}/loopback/loopback.xo)

set(ACCL_UDP_XOS ${ACCL_PLUGINS_LOOPBACK} ${ACCL_PLUGINS_COMPRESSION} ${ACCL_PLUGINS_SUM} ${ACCL_PLUGINS_HOSTCTRL}
    ${ACCL_CCLO_KERNEL_XO} ${ACCL_UDP_MAC_XO} ${ACCL_UDP_NET_XO} PARENT_SCOPE)

add_custom_target(
    accl_udp_stack
    DEPENDS ${ACCL_UDP_MAC_XO} ${ACCL_UDP_NET_XO})

add_custom_target(
    accl_cclo
    DEPENDS ${ACCL_CCLO_KERNEL_DIR}/${ACCL_CCLO_KERNEL_XO})

add_custom_command(
    OUTPUT ${ACCL_PLUGINS_HOSTCTRL}
    COMMAND vitis_hls build_hostctrl.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/hostctrl ) 
add_custom_command(
    OUTPUT ${ACCL_PLUGINS_SUM}
    COMMAND vitis_hls build.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/reduce_sum ) 
add_custom_command(
    OUTPUT ${ACCL_PLUGINS_COMPRESSION}
    COMMAND vitis_hls build.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/hp_compression ) 
add_custom_command(
    OUTPUT ${ACCL_PLUGINS_LOOPBACK}
    COMMAND vitis_hls build_loopback.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/loopback ) 

add_custom_target(
    accl_plugins
    DEPENDS ${ACCL_PLUGINS_LOOPBACK} ${ACCL_PLUGINS_SUM} ${ACCL_PLUGINS_HOSTCTRL} 
    ${ACCL_PLUGINS_COMPRESSION})

add_custom_target(
    accl_udp)
add_dependencies(accl_udp accl_udp_stack accl_cclo accl_plugins)

