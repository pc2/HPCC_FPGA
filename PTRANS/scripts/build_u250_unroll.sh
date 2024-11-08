#!/bin/bash

SCRIPT_PATH=${PWD}

BENCHMARK_DIR=${SCRIPT_PATH}/../

SYNTH_DIR=/mnt/local/meyermar/synth/u250/PTRANS

CONFIG_NAMES=("Xilinx_U250_DDR_PCIE_unroll")

for r in "${CONFIG_NAMES[@]}"; do
    BUILD_DIR=${SYNTH_DIR}/${r}

    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ${BENCHMARK_DIR} -DCMAKE_BUILD_TYPE=Release -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/${r}.cmake

    make transpose_PQ_PCIE_xilinx Transpose_xilinx
done
