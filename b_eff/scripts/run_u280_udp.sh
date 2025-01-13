#!/bin/bash
#SBATCH -p fpga
#SBATCH --constraint=xilinx_u280_xrt2.14
#SBATCH -t 2:00:00
#SBATCH --mem=10g
#SBATCH -o beff_udp_n2-%j.out
#SBATCH -e beff_udp_n2-%j.out
#SBATCH -n 2
#SBATCH -N 1

module reset
module load devel toolchain lib fpga foss Boost xilinx/xrt/2.14 changeFPGAlinks CMake

xbutil reset -d 0000:a1:00.1 --force
xbutil reset -d 0000:81:00.1 --force
xbutil reset -d 0000:01:00.1 --force

changeFPGAlinks --fpgalink=n00:acl0:ch0-n00:acl1:ch0

# first run may fail to find a link
srun -n 2 ./build_u280_udp/bin/Network_xilinx -f ./build_u280_udp/bin/communication_UDP.xclbin -r 1 -l 64 --min-size 6 -m 28 -d 14 --payload-size 7 --dump-json beff_udp_7_n2-$SLURM_JOBID.json
srun -n 2 ./build_u280_udp/bin/Network_xilinx -f ./build_u280_udp/bin/communication_UDP.xclbin -r 1 -l 64 --min-size 6 -m 28 -d 14 --payload-size 5 --dump-json beff_udp_5_n2-$SLURM_JOBID.json
srun -n 2 ./build_u280_udp/bin/Network_xilinx -f ./build_u280_udp/bin/communication_UDP.xclbin -r 1 -l 64 --min-size 6 -m 28 -d 14 --payload-size 4 --dump-json beff_udp_4_n2-$SLURM_JOBID.json
srun -n 2 ./build_u280_udp/bin/Network_xilinx -f ./build_u280_udp/bin/communication_UDP.xclbin -r 1 -l 64 --min-size 6 -m 28 -d 14 --payload-size 2 --dump-json beff_udp_2_n2-$SLURM_JOBID.json
