#!/bin/bash
#SBATCH -p fpgasyn

module load devel/CMake
module load intelFPGA_pro/19.4.0
module load nalla_pcie/19.2.0_hpc

LOG_FFT_SIZE=12


BUILD_DIR=build-${QUARTUS_VERSION}-${QUARTUS_VERSION_BSP}-${LOG_FFT_SIZE}

mkdir ../${BUILD_DIR}
cd ../${BUILD_DIR}

cmake .. -DLOG_FFT_SIZE=${LOG_FFT_SIZE} -DFPGA_BOARD_NAME=${FPGA_BOARD_NAME}

make fft1d_float_8


