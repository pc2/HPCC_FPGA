#!/bin/bash
#
# Execute the matrix transposition with 9 FPGAs
# Sets up the network topology for the execution with 9 FPGAs with FPGAs on the same node being pairs and the
# 9th FPGA calculating the diagonal blocks.
#
#SBATCH -p fpga
#SBATCH --constraint=19.4.0_max
#SBATCH -N 5
#SBATCH -n 9
# 
# Diagonal node
#
#SBATCH --fpgalink="n04:acl0:ch0-n04:acl0:ch1"
#SBATCH --fpgalink="n04:acl0:ch2-n04:acl0:ch3"
#
# Other node pairs
#
#SBATCH --fpgalink="n00:acl0:ch0-n00:acl1:ch1"
#SBATCH --fpgalink="n00:acl0:ch2-n00:acl1:ch3"
#SBATCH --fpgalink="n00:acl0:ch1-n00:acl1:ch0"
#SBATCH --fpgalink="n00:acl0:ch3-n00:acl1:ch2"
#SBATCH --fpgalink="n01:acl0:ch0-n01:acl1:ch1"
#SBATCH --fpgalink="n01:acl0:ch2-n01:acl1:ch3"
#SBATCH --fpgalink="n01:acl0:ch1-n01:acl1:ch0"
#SBATCH --fpgalink="n01:acl0:ch3-n01:acl1:ch2"
#SBATCH --fpgalink="n02:acl0:ch0-n02:acl1:ch1"
#SBATCH --fpgalink="n02:acl0:ch2-n02:acl1:ch3"
#SBATCH --fpgalink="n02:acl0:ch1-n02:acl1:ch0"
#SBATCH --fpgalink="n02:acl0:ch3-n02:acl1:ch2"
#SBATCH --fpgalink="n03:acl0:ch0-n03:acl1:ch1"
#SBATCH --fpgalink="n03:acl0:ch2-n03:acl1:ch3"
#SBATCH --fpgalink="n03:acl0:ch1-n03:acl1:ch0"
#SBATCH --fpgalink="n03:acl0:ch3-n03:acl1:ch2"

module load intelFPGA_pro/20.3.0
module load bittware_520n/19.4.0_max
module load intel
module load devel/CMake/3.15.3-GCCcore-8.3.0

# Path to the host executable
BIN_FILE=/scratch/pc2-mitarbeiter/mariusme/devel/HPCC_FPGA_ptrans/build/520n/PTRANS/bin/Transpose_intel

# Path to the bitstream
AOCX_FILE=$PFS_SCRATCH/synth/520n/PTRANS/20.3.0-19.4.0_max-Nallatech_520N-noloop/bin/transpose_diagonal.aocx

# Execute the benchmark with matrices of 2GB size each 
# -m 48 width of 48 blocks of 512 data items of SP floating point = 2GB
# -n 10 repeat measurement 10 times
# -r 4 use 4 kernel replications on every FPGA
srun ${BIN_FILE} -f ${AOCX_FILE} -n 10 -m 48 -r 4
