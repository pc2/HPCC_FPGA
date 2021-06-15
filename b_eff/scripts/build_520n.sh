#!/bin/bash
#
# Synthesize the b_eff kernel for the Nallaatech 520N board.
# This is an example script, how the synthesis can be started on Noctua using a HPCC FPGA configuration file.
# Submit this script to sbatch in this folder!
#
#SBATCH -p fpgasyn
#SBATCH -J b_eff

module load intelFPGA_pro/20.4.0
module load nalla_pcie/19.4.0_hpc
module load intel
module load devel/CMake/3.15.3-GCCcore-8.3.0

SCRIPT_PATH=${SLURM_SUBMIT_DIR}

BENCHMARK_DIR=${SCRIPT_PATH}/../

SYNTH_DIR=${PFS_SCRATCH}/synth/520n/multi_fpga/b_eff


mkdir -p ${SYNTH_DIR}
cd ${SYNTH_DIR}

cmake ${BENCHMARK_DIR} -DCMAKE_BUILD_TYPE=Release -DHPCC_FPGA_CONFIG=${BENCHMARK_DIR}/configs/Bittware_520N.cmake

make communication_bw520n_intel Network_intel

