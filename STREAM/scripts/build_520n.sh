#!/bin/bash
#
# Synthesize the STREAM single kernel for the Nallaatech 520N board.
# This is an example script, how the synthesis can be started on Noctua using a HPCC FPGA configuration file.
# Submit this script to sbatch in this folder!
#
#SBATCH -p fpgasyn
#SBATCH --exclusive

module load intelFPGA_pro/19.4.0
module load nalla_pcie/19.4.0_hpc
module load intel
module load devel/CMake/3.15.3-GCCcore-8.3.0

SCRIPT_PATH=${SLURM_SUBMIT_DIR}

BENCHMARK_DIR=${SCRIPT_PATH}/../

SYNTH_DIR=${PFS_SCRATCH}/synth/520n/STREAM

for r in {1..3}; do
    BUILD_HP=${SYNTH_DIR}/Nallatech_520N_HP_${r}
    BUILD_SP=${SYNTH_DIR}/Nallatech_520N_SP_${r}
    BUILD_DP=${SYNTH_DIR}/Nallatech_520N_DP_${r}

    mkdir -p ${BUILD_HP}
    cd ${BUILD_HP}

    cmake ${BENCHMARK_DIR} -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/Nallatech_520N_HP.cmake

    make stream_kernels_single_intel&

    mkdir -p ${BUILD_SP}
    cd ${BUILD_SP}

    cmake ${BENCHMARK_DIR} -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/Nallatech_520N_SP.cmake

    make stream_kernels_single_intel&

    mkdir -p ${BUILD_DP}
    cd ${BUILD_DP}

    cmake ${BENCHMARK_DIR} -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/Nallatech_520N_DP.cmake

    make stream_kernels_single_intel&

    wait
done
