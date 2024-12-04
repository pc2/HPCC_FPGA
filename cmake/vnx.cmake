set(VNX_UDP_ETH_IFS 1 CACHE STRING "Number of Ethernet interfaces to synthesize for UDP stack")
# Jumbo frames are 128 * 64 Byte = 8192 Byte
# Clode to default frame size is 16 * 64 Byte = 1024 Byte
set(VNX_MAX_FRAME_SIZE 7 CACHE STRING "Maximum used frame size in chunks of 64Byte. Specified as exponent of 2!")
set(VNX_UDP_NET_XO ${extern_vnx_udp_SOURCE_DIR}/NetLayers/_x.${FPGA_BOARD_NAME}/networklayer.xo)
set(UDP_HLS_IP_FOLDER ${extern_vnx_udp_SOURCE_DIR}/NetLayers/100G-fpga-network-stack-core/synthesis_results_HBM)
set(UDP_HLS_SRC_DIR ${extern_vnx_udp_SOURCE_DIR}/NetLayers/100G-fpga-network-stack-core)

list(APPEND UDP_LINK_CONFIG --advanced.param compiler.userPostSysLinkOverlayTcl=${extern_vnx_udp_SOURCE_DIR}/Ethernet/post_sys_link.tcl)
list(APPEND UDP_LINK_CONFIG --user_ip_repo_paths ${UDP_HLS_IP_FOLDER})

list(APPEND XILINX_ADDITIONAL_COMPILE_FLAGS "-I${extern_accl_SOURCE_DIR}/driver/hls" "-DACCL_SYNTHESIS")

set(VNX_UDP_MAC_XOS "")

math(EXPR loopend "${VNX_UDP_ETH_IFS} - 1")
foreach(i RANGE ${loopend})
    set(CURRENT_MAC_XO ${extern_vnx_udp_SOURCE_DIR}/Ethernet/_x.${FPGA_BOARD_NAME}/cmac_${i}.xo)
    add_custom_command(
        OUTPUT ${CURRENT_MAC_XO}
        COMMAND make -C ${extern_vnx_udp_SOURCE_DIR}/Ethernet DEVICE=${FPGA_BOARD_NAME} INTERFACE=${i} all
        WORKING_DIRECTORY ${extern_vnx_udp_SOURCE_DIR}) 
    list(APPEND VNX_UDP_MAC_XOS ${CURRENT_MAC_XO})
endforeach()

add_custom_command(
    OUTPUT ${VNX_UDP_NET_XO}
    COMMAND make DEVICE=${FPGA_BOARD_NAME} MAX_SOCKETS=63 NetLayers/_x.${FPGA_BOARD_NAME}/networklayer.xo
    WORKING_DIRECTORY ${extern_vnx_udp_SOURCE_DIR}) 

add_custom_target(
    vnx_udp_stack
    DEPENDS ${VNX_UDP_MAC_XOS} ${VNX_UDP_NET_XO})

set(UDP_XOS ${VNX_UDP_MAC_XOS} ${VNX_UDP_NET_XO} CACHE INTERNAL "Object files required for VNx UDP")