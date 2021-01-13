#!/bin/bash
#
# Synthesize GEMM for the BittWare 520N MX board.
# This is an example script, how the synthesis can be started on Noctua using a HPCC FPGA configuration file.
# Submit this script to sbatch in this folder!
#
#SBATCH -p fpgasyn

module load intelFPGA_pro/20.3.0
module load bittware_520n_mx/19.4.0_hpc
module load intel
module load devel/CMake/3.15.3-GCCcore-8.3.0

SCRIPT_PATH=${SLURM_SUBMIT_DIR}

BENCHMARK_DIR=${SCRIPT_PATH}/../

SYNTH_DIR=${PFS_SCRATCH}/synth/520nmx/GEMM

CONFIG_NAMES=("Bittware_520N_MX_B256")

for r in "${CONFIG_NAMES[@]}"; do
    BUILD_DIR=${SYNTH_DIR}/20.3.0-19.4.0-${r}

    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ${BENCHMARK_DIR} -DCMAKE_BUILD_TYPE=Release -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/${r}.cmake

    make gemm_base_intel GEMM_intel
done
