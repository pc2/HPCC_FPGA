#!/bin/bash
#
# Synthesize the STREAM single kernel for the Stratix 10 MX HBM board on Noctua.
# Submit this script to sbatch in this folder!
#
#SBATCH -p fpgasyn
#SBATCH --exclusive

module load intelFPGA_pro/20.4.0
module load bittware_520n_mx/19.4.0_hpc
module load intel
module load devel/CMake/3.15.3-GCCcore-8.3.0

SCRIPT_PATH=${SLURM_SUBMIT_DIR}

BENCHMARK_DIR=${SCRIPT_PATH}/../

SYNTH_DIR=${PFS_SCRATCH}/synth/520nmx/STREAM

CONFIG_NAMES=("Bittware_520N_MX_SP_256bit_small")

for r in "${CONFIG_NAMES[@]}"; do
    BUILD_DIR=${SYNTH_DIR}/20.4.0-19.4.0-${r}-nofusion

    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ${BENCHMARK_DIR} -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/${r}.cmake

    make stream_kernels_single_intel STREAM_FPGA_intel&
done

wait

