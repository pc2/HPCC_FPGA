#!/bin/bash
#
# Run the b_eff kernel for the u280 board using HBM over PCIE.
# This is an example script, how the synthesis can be started on Noctua using a HPCC FPGA configuration file.
# Submit this script from the benchmark folder!
#
#SBATCH -p fpga
#SBATCH --constraint=xilinx_u280_xrt2.14
#SBATCH -t 2:00:00
#SBATCH --mem=10g
#SBATCH -o beff_pcie_n2-%j.out
#SBATCH -e beff_pcie_n2-%j.out
#SBATCH -n 2
#SBATCH -N 1

module reset
module load devel toolchain lib fpga foss Boost xilinx/xrt/2.14 changeFPGAlinks CMake

srun -n 2 build_u280_pcie_hbm/bin/Network_xilinx -f build_u280_pcie_hbm/bin/communication_PCIE.xclbin --comm-type PCIE
