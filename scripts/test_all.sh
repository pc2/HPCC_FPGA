#!/bin/bash
#
# This script builds and tests every benchmark with the default configuration.
# It will create a folder build/test in the root folder of the project and place
# additional logs for build and test there.
# To start a clean build, just delete the folder an re-run the script.
# The run is successfull, if the script returns with an error code of 0.
# If an error occured during build or test, the script will stop immediately.
# The script uses a reduced -DBLOCK_SIZE=32 to speed up the tests for LINPACK
# and GEMM.
# Additional build configurations can be given as command line option:
# i.e.:
#     ./test_all.sh -DFPGA_BOARD_NAME=other_board
#


SCRIPT_PATH=$( cd "$(dirname $0)"; pwd -P)

PROJECT_ROOT=${SCRIPT_PATH}/..

TEST_DIR=${PROJECT_ROOT}/build/test

BUILD_LOG_FILE=${TEST_DIR}/lastbuild.log
TEST_LOG_FILE=${TEST_DIR}/lasttests.log

BENCHMARKS=("b_eff" "FFT" "GEMM" "LINPACK" "PTRANS" "RandomAccess" "STREAM")

mkdir -p $TEST_DIR
rm -f $BUILD_LOG_FILE
rm -f $TEST_LOG_FILE

cd $PROJECT_ROOT

echo "Updating git submodules..."
git submodule update --init --recursive

cd $TEST_DIR

echo "Start building hosts code, tests and emulation kernel for all benchmarks."

for bm in ${BENCHMARKS[@]}; do
    echo "Building $bm..."
    cd $TEST_DIR
    mkdir -p $bm
    ret=0
    cd $bm
    cmake ${PROJECT_ROOT}/$bm -DDEFAULT_DEVICE=0 -DDEFAULT_PLATFORM=0 -DBLOCK_SIZE=32 $@ &>> $BUILD_LOG_FILE
    ret=$(($ret + $?))
    make all &>> $BUILD_LOG_FILE
    ret=$(($ret + $?))
    if [ $ret -ne 0 ]; then
        echo "Failed building $bm"
        exit $ret
    fi
done

echo "Start testing all benchmarks"

for bm in ${BENCHMARKS[@]}; do
    echo "Testing $bm..."
    cd $TEST_DIR
    ret=0
    cd $bm
    if [ $bm = "b_eff" ]; then
        cd bin
        touch kernel_output_ch0
        touch kernel_output_ch1
        touch kernel_output_ch2
        touch kernel_output_ch3
        ln -s kernel_output_ch0 kernel_input_ch1
        ln -s kernel_output_ch2 kernel_input_ch3
        ln -s kernel_output_ch1 kernel_input_ch0
        ln -s kernel_output_ch3 kernel_input_ch2
        cd ..
    fi
    make CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 CTEST_OUTPUT_ON_FAILURE=1 test &>> $TEST_LOG_FILE
    ret=$(($ret + $?))
    if [ $ret -ne 0 ]; then
        echo "Failed testing $bm"
        exit $ret
    fi
done

echo "-----------"
echo "SUCCESS!!!!"
echo "-----------"
