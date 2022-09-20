
# General definitions
set(ACCL_STACK_TYPE "UDP" CACHE STRING "Network layer used in ACCL")
set(ACCL_UDP_ETH_IFS 1 CACHE STRING "Number of Ethernet interfaces to synthesize for UDP stack")
set(ACCL_DEVICE_NAME "xcu280-fsvh2892-2L-e" CACHE STRING "Name of the FPGA used on the target platform")
set(ACCL_BUFFER_SIZE 8192 CACHE STRING "Size of ACCL buffers in bytes")
set(ACCL_HARDWARE_DIR ${extern_accl_SOURCE_DIR}/test/hardware)
set(ACCL_CCLO_ADDITIONAL_BUILD_ARGS "" CACHE STRING "Add additional build arguments that will be passed to the CCLO makefile")
set(ACCL_CCLO_BUILD_ARGS ${ACCL_CCLO_ADDITIONAL_BUILD_ARGS})
# UDP related definitions
set(ACCL_VNX_DIR ${ACCL_HARDWARE_DIR}/xup_vitis_network_example/)
set(ACCL_NETLAYER_HLS ${ACCL_VNX_DIR}/NetLayers/100G-fpga-network-stack-core)
set(ACCL_UDP_NET_XO ${ACCL_VNX_DIR}/NetLayers/_x.${FPGA_BOARD_NAME}/networklayer.xo)
set(ACCL_HLS_IP_FOLDER ${ACCL_NETLAYER_HLS}/synthesis_results_HBM)
if (ACCL_STACK_TYPE STREQUAL "UDP")
    list(APPEND ACCL_LINK_CONFIG --advanced.param compiler.userPostSysLinkOverlayTcl=${ACCL_VNX_DIR}/Ethernet/post_sys_link.tcl)
    list(APPEND ACCL_LINK_CONFIG --user_ip_repo_paths ${ACCL_HLS_IP_FOLDER})
    list(APPEND ACCL_CCLO_BUILD_ARGS STACK_TYPE=${ACCL_STACK_TYPE})
endif()

set(ACCL_UDP_MAC_XOS "")

math(EXPR loopend "${ACCL_UDP_ETH_IFS} - 1")
foreach(i RANGE ${loopend})
    set(CURRENT_MAC_XO ${ACCL_VNX_DIR}/Ethernet/_x.${FPGA_BOARD_NAME}/cmac_${i}.xo)
    add_custom_command(
        OUTPUT ${CURRENT_MAC_XO}
        COMMAND make -C ${ACCL_VNX_DIR}/Ethernet DEVICE=${FPGA_BOARD_NAME} INTERFACE=${i} all
        WORKING_DIRECTORY ${ACCL_HARDWARE_DIR}) 
    list(APPEND ACCL_UDP_MAC_XOS ${CURRENT_MAC_XO})
endforeach()

add_custom_command(
    OUTPUT ${ACCL_UDP_NET_XO}
    COMMAND make -C ${ACCL_VNX_DIR}/NetLayers DEVICE=${FPGA_BOARD_NAME} all
    WORKING_DIRECTORY ${ACCL_HARDWARE_DIR}) 

add_custom_target(
    accl_udp_stack
    DEPENDS ${ACCL_UDP_MAC_XOS} ${ACCL_UDP_NET_XO})

# TCP related definitions
set(ACCL_TCP_BASE_DIR ${ACCL_HARDWARE_DIR}/Vitis_with_100Gbps_TCP-IP)
set(ACCL_TCP_XO ${ACCL_TCP_BASE_DIR}/_x.hw.${FPGA_BOARD_NAME}/network_krnl.xo)
set(ACCL_TCP_CMAC_XO ${ACCL_TCP_BASE_DIR}/_x.hw.${FPGA_BOARD_NAME}/cmac_krnl.xo)
if (ACCL_STACK_TYPE STREQUAL "TCP")
    list(APPEND ACCL_LINK_CONFIG --advanced.param compiler.userPostSysLinkOverlayTcl=${ACCL_TCP_BASE_DIR}/scripts/post_sys_link.tcl)
    list(APPEND ACCL_LINK_CONFIG --user_ip_repo_paths ${ACCL_TCP_BASE_DIR}/build/fpga-network-stack/iprepo)
    list(APPEND ACCL_CCLO_BUILD_ARGS STACK_TYPE=${ACCL_STACK_TYPE} EN_FANIN=1)
endif()

# TODO: This is very sppecific to the Xilinx build system, because
# different Vivado version is required to build these ips
add_custom_command(
    OUTPUT ${ACCL_TCP_BASE_DIR}/build/fpga-network-stack/iprepo
    COMMAND mkdir build && cd build && cmake .. -DFDEV_NAME=u280 
            -DVIVADO_HLS_ROOT_DIR=/proj/xbuilds/2020.1_released/installs/lin64/Vivado/2020.1 
            -DVIVADO_ROOT_DIR=/proj/xbuilds/2020.1_released/installs/lin64/Vivado/2020.1 
            -DTCP_STACK_EN=1 -DTCP_STACK_RX_DDR_BYPASS_EN=1 -DTCP_STACK_WINDOW_SCALING_EN=0 &&
            make installip
    WORKING_DIRECTORY ${ACCL_TCP_BASE_DIR})

add_custom_command(
    OUTPUT ${ACCL_TCP_CMAC_XO}
    COMMAND make cmac_krnl DEVICE=${FPGA_BOARD_NAME} XSA=${FPGA_BOARD_NAME} TEMP_DIR=_x.hw.${FPGA_BOARD_NAME}/
    WORKING_DIRECTORY ${ACCL_TCP_BASE_DIR}
    DEPENDS ${ACCL_TCP_BASE_DIR}/build/fpga-network-stack/iprepo) 

add_custom_command(
    OUTPUT ${ACCL_TCP_XO}
    COMMAND make network_krnl DEVICE=${FPGA_BOARD_NAME} XSA=${FPGA_BOARD_NAME} TEMP_DIR=_x.hw.${FPGA_BOARD_NAME}/
    WORKING_DIRECTORY ${ACCL_TCP_BASE_DIR}
    DEPENDS ${ACCL_TCP_BASE_DIR}/build/fpga-network-stack/iprepo) 

add_custom_target(
    accl_tcp_stack
    DEPENDS ${ACCL_TCP_XO} ${ACCL_TCP_CMAC_XO})
      

# Build CCLO
set(ACCL_CCLO_KERNEL_DIR ${extern_accl_SOURCE_DIR}/kernels/cclo/)
set(ACCL_CCLO_KERNEL_XO ccl_offload.xo)

add_custom_command(
    OUTPUT ${ACCL_CCLO_KERNEL_DIR}/${ACCL_CCLO_KERNEL_XO}
    COMMAND make ${ACCL_CCLO_BUILD_ARGS} PLATFORM=${FPGA_BOARD_NAME}
    WORKING_DIRECTORY ${ACCL_CCLO_KERNEL_DIR})

add_custom_target(
    accl_cclo
    DEPENDS ${ACCL_CCLO_KERNEL_DIR}/${ACCL_CCLO_KERNEL_XO})

# Build the ACCL Plugins
set(ACCL_PLUGINS_DIR ${extern_accl_SOURCE_DIR}/kernels/plugins)
set(ACCL_PLUGINS_HOSTCTRL ${ACCL_PLUGINS_DIR}/hostctrl/hostctrl.xo)
set(ACCL_PLUGINS_SUM ${ACCL_PLUGINS_DIR}/reduce_ops/reduce_ops.xo)
set(ACCL_PLUGINS_COMPRESSION ${ACCL_PLUGINS_DIR}/hp_compression/hp_compression.xo)
set(ACCL_PLUGINS_LOOPBACK ${ACCL_PLUGINS_DIR}/loopback/loopback.xo)
set(ACCL_PLUGINS_ARBITER ${ACCL_PLUGINS_DIR}/client_arbiter/client_arbiter.xo)

add_custom_command(
    OUTPUT ${ACCL_PLUGINS_HOSTCTRL}
    COMMAND vitis_hls build_hostctrl.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/hostctrl ) 
add_custom_command(
    OUTPUT ${ACCL_PLUGINS_SUM}
    COMMAND vitis_hls build.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/reduce_ops ) 
add_custom_command(
    OUTPUT ${ACCL_PLUGINS_COMPRESSION}
    COMMAND vitis_hls build.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/hp_compression ) 
add_custom_command(
    OUTPUT ${ACCL_PLUGINS_LOOPBACK}
    COMMAND vitis_hls build_loopback.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/loopback ) 
add_custom_command(
    OUTPUT ${ACCL_PLUGINS_ARBITER}
    COMMAND vitis_hls build_client_arbiter.tcl -tclargs ip ${ACCL_DEVICE_NAME}
    WORKING_DIRECTORY ${ACCL_PLUGINS_DIR}/client_arbiter ) 


add_custom_target(
    accl_plugins
    DEPENDS ${ACCL_PLUGINS_LOOPBACK} ${ACCL_PLUGINS_SUM} ${ACCL_PLUGINS_HOSTCTRL} 
    ${ACCL_PLUGINS_COMPRESSION} ${ACCL_PLUGINS_ARBITER})

set(ACCL_UDP_XOS ${ACCL_PLUGINS_LOOPBACK} ${ACCL_PLUGINS_COMPRESSION} ${ACCL_PLUGINS_SUM} ${ACCL_PLUGINS_HOSTCTRL}
    ${ACCL_CCLO_KERNEL_DIR}/${ACCL_CCLO_KERNEL_XO} ${ACCL_UDP_MAC_XOS} ${ACCL_UDP_NET_XO} CACHE INTERNAL "Object files required for ACCL with UDP")

set(ACCL_TCP_XOS ${ACCL_PLUGINS_LOOPBACK} ${ACCL_PLUGINS_COMPRESSION} ${ACCL_PLUGINS_SUM} ${ACCL_PLUGINS_HOSTCTRL}
    ${ACCL_CCLO_KERNEL_DIR}/${ACCL_CCLO_KERNEL_XO} ${ACCL_TCP_CMAC_XO} ${ACCL_TCP_XO} CACHE INTERNAL "Object files required for ACCL with TCP")

if (DEFINED USE_ACCL_CLIENT_ARBITER)
    list(APPEND ${ACCL_UDP_XOS} ${ACCL_PLUGINS_ARBITER})
    list(APPEND ${ACCL_TCP_XOS} ${ACCL_PLUGINS_ARBITER})
endif()
if (ACCL_STACK_TYPE STREQUAL "UDP")
    set(ACCL_XOS ${ACCL_UDP_XOS} CACHE INTERNAL "Object files required for ACCL")
else()
    set(ACCL_XOS ${ACCL_TCP_XOS} CACHE INTERNAL "Object files required for ACCL")
endif()

add_custom_target(
    accl_udp)
add_dependencies(accl_udp accl_udp_stack accl_cclo accl_plugins)

add_custom_target(
    accl_tcp)
add_dependencies(accl_tcp accl_tcp_stack accl_cclo accl_plugins)

add_custom_target(accl_device)
if (ACCL_STACK_TYPE STREQUAL "UDP")
    add_dependencies(accl_device accl_udp)
else()
    add_dependencies(accl_device accl_tcp)
endif()
