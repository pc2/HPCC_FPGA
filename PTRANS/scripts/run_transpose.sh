#!/bin/bash
#
# Execute the matrix transposition with a single FPGA
# Sets up the network topology for the execution as a loopback.
#
#SBATCH -p fpga
#SBATCH --constraint=19.4.0_max
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --fpgalink="n00:acl0:ch0-n00:acl0:ch0"
#SBATCH --fpgalink="n00:acl0:ch1-n00:acl0:ch1"
#SBATCH --fpgalink="n00:acl0:ch2-n00:acl0:ch2"
#SBATCH --fpgalink="n00:acl0:ch3-n00:acl0:ch3"


module load intelFPGA_pro/20.3.0
module load nalla_pcie/19.4.0_max
module load intel
module load devel/CMake/3.15.3-GCCcore-8.3.0

BIN_FILE=$PFS_SCRATCH/synth/pac/LINPACK/20.3.0-19.4.0_max-Nallatech_520N/bin/Transpose_intel
AOCX_FILE=$PFS_SCRATCH/synth/pac/LINPACK/20.3.0-19.4.0_max-Nallatech_520N/bin/transpose_diagonal.aocx

srun ${BIN_FILE} -f ${AOCX_FILE} -n 1 -m 1
