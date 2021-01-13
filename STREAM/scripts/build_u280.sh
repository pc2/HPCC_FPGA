#!/bin/bash

BENCHMARK_DIR=${PWD}/../

SYNTH_DIR=~/synth/u280/STREAM

for r in {1..3}; do
    BUILD_HP=${SYNTH_DIR}/Xilinx_U280_HP_${r}
    BUILD_SP=${SYNTH_DIR}/Xilinx_U280_SP_${r}
    BUILD_DP=${SYNTH_DIR}/Xilinx_U280_DP_${r}

    mkdir -p ${BUILD_HP}
    cd ${BUILD_HP}

    cmake ${BENCHMARK_DIR} -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/Xilinx_U280_HP.cmake

    make stream_kernels_single_xilinx

    mkdir -p ${BUILD_SP}
    cd ${BUILD_SP}

    cmake ${BENCHMARK_DIR} -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/Xilinx_U280_SP.cmake

    make stream_kernels_single_xilinx

    mkdir -p ${BUILD_DP}
    cd ${BUILD_DP}

    cmake ${BENCHMARK_DIR} -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/Xilinx_U280_DP.cmake

    make stream_kernels_single_xilinx

done
