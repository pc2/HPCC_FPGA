#!/bin/bash
#
# Execute the matrix transposition with N FPGAs.
# This is a generator script. Use it as input for the code generator.
# Example to generate script for execution on 19 FPGAs:
#
#    ../../scripts/code_generator/generator.py --comment="#" --comment-ml-start="$" --comment-ml-end="$" run_transpose_gen.sh -o run_transpose_n19.sh
#
# You can specify the number of used nodes with the parameter N.
# To generate a script for the execution on 9 FPGAs:
#
#    ../../scripts/code_generator/generator.py -p "N=5" --comment="#" --comment-ml-start="$" --comment-ml-end="$" run_transpose_gen.sh -o run_transpose_n9.sh
#
# PY_CODE_GEN N=10
# PY_CODE_GEN block_start replace()
#SBATCH -p fpga
#SBATCH --constraint=19.4.0_max
#SBATCH -N $PY_CODE_GEN N$
#SBATCH -n $PY_CODE_GEN N*2 - 1$
# 
# Diagonal node
#
#SBATCH --fpgalink="n$PY_CODE_GEN f"{N-1:02d}"$:acl0:ch0-n$PY_CODE_GEN f"{N-1:02d}"$:acl0:ch1"
#SBATCH --fpgalink="n$PY_CODE_GEN f"{N-1:02d}"$:acl0:ch2-n$PY_CODE_GEN f"{N-1:02d}"$:acl0:ch3"
# PY_CODE_GEN block_end
#
# Other node pairs
#
# PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(N - 1)]
#SBATCH --fpgalink="n$PY_CODE_GEN f"{i:02d}"$:acl0:ch0-n$PY_CODE_GEN f"{i:02d}"$:acl1:ch1"
#SBATCH --fpgalink="n$PY_CODE_GEN f"{i:02d}"$:acl0:ch2-n$PY_CODE_GEN f"{i:02d}"$:acl1:ch3"
#SBATCH --fpgalink="n$PY_CODE_GEN f"{i:02d}"$:acl0:ch1-n$PY_CODE_GEN f"{i:02d}"$:acl1:ch0"
#SBATCH --fpgalink="n$PY_CODE_GEN f"{i:02d}"$:acl0:ch3-n$PY_CODE_GEN f"{i:02d}"$:acl1:ch2"
#PY_CODE_GEN block_end


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
srun ${BIN_FILE} -f ${AOCX_FILE} -n 10 -m 128 -r 4
