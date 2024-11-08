#!/bin/bash
#
# Synthesize the PTRANS kernel for the Nallatech 520N board.
# This is an example script, how the synthesis can be started on Noctua using a HPCC FPGA configuration file.
# Submit this script to sbatch in this folder!
#
#SBATCH -p fpgasyn
#SBATCH -J PTRANS

module load intelFPGA_pro/20.4.0
module load nalla_pcie/19.4.0_hpc
module load intel
module load devel/CMake/3.15.3-GCCcore-8.3.0

SCRIPT_PATH=${SLURM_SUBMIT_DIR}

BENCHMARK_DIR=${SCRIPT_PATH}/../

SYNTH_DIR=${PFS_SCRATCH}/synth/520n/multi_fpga/PTRANS/pq_pcie

CONFIG_NAMES=("Nallatech_520N_pcie")

for r in "${CONFIG_NAMES[@]}"; do
    BUILD_DIR=${SYNTH_DIR}/20.4.0-19.4.0_hpc-${r}

    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}

    cmake ${BENCHMARK_DIR} -DCMAKE_BUILD_TYPE=Release -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/${r}.cmake

    make transpose_PQ_PCIE_intel Transpose_intel
done
