#!/bin/bash
#
# Synthesize the STREAM single kernel for the Stratix 10 MX HBM board on Noctua.
# Submit this script to sbatch in this folder!
#
#SBATCH -p fpgasyn

module load intelFPGA_pro/19.4.0
module load intel_s10mx/19.3.0
module load lang/Python/3.7.0-foss-2018b
module load devel/CMake/3.15.3-GCCcore-8.3.0

SCRIPT_PATH=${SLURM_SUBMIT_DIR}

BENCHMARK_DIR=${SCRIPT_PATH}/../

BUILD_DIR_4K=${SCRIPT_PATH}/../../build/synth/RA-s10xm_hbm

mkdir -p ${BUILD_DIR_4K}
cd ${BUILD_DIR_4K}

cmake ${BENCHMARK_DIR} -DDEVICE_BUFFER_SIZE=1024 -DNUM_REPLICATIONS=32 \
        -DAOC_FLAGS="-fpc -fp-relaxed -global-ring" \
            -DINTEL_CODE_GENERATION_SETTINGS=${BENCHMARK_DIR}/settings/settings.gen.intel.random_access_kernels_single.s10mxhbm.py

make random_access_kernels_single_intel&

wait

