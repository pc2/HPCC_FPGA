#!/bin/bash
#
# This script builds and tests every benchmark with the default configuration.
# It will create a folder build/test in the root folder of the project and place
# additional logs for build and test there.
# To start a clean build, just delete the folder an re-run the script.
# The run is successfull, if the script returns with an error code of 0.
# If an error occured during build or test, the script will stop immediately.
# The script uses a reduced -DBLOCK_SIZE=32 to speed up the tests for GEMM.
# Additional build configurations can be given as command line option:
# i.e.:
#     ./test_all.sh -DFPGA_BOARD_NAME=other_board
#

SCRIPT_PATH=$( cd "$(dirname $0)"; pwd -P)

PROJECT_ROOT=${SCRIPT_PATH}/..

TEST_DIR=${PROJECT_ROOT}/build/test

BUILD_LOG_FILE=${TEST_DIR}/lastbuild.log
TEST_LOG_FILE=${TEST_DIR}/lasttests.log

BENCHMARKS=("b_eff" "LINPACK" "PTRANS")
#BENCHMARKS=("b_eff" "FFT" "GEMM" "LINPACK" "PTRANS" "RandomAccess" "STREAM")
if [ "$1" != "inc" ]; then
    echo "Clean build directory, use option 'inc' to prevent this!"
    rm -rf ${TEST_DIR}
else
    echo "Do incremental build based on previous run!"
fi

mkdir -p $TEST_DIR
rm -f $BUILD_LOG_FILE
rm -f $TEST_LOG_FILE

cd $TEST_DIR

echo "Start building hosts code, tests and emulation kernel for all benchmarks."

for bm in ${BENCHMARKS[@]}; do
    echo "Building $bm..."
    if [ -f  ${TEST_DIR}/$bm/BUILD_SUCCESS ]; then
        continue
    else
        rm -rf ${TEST_DIR}/$bm
    fi
    cd $TEST_DIR
    mkdir -p $bm
    ret=0
    cd $bm
    cmake ${PROJECT_ROOT}/$bm -DUSE_OCL_HOST=Yes -DUSE_DEPRECATED_HPP_HEADER=Yes -DDEFAULT_DEVICE=0 -DDEFAULT_PLATFORM=0 -DBLOCK_SIZE=32 &>> $BUILD_LOG_FILE
    ret=$(($ret + $?))
    make -j 40 VERBOSE=1 all &>> $BUILD_LOG_FILE
    ret=$(($ret + $?))
    if [ $ret -ne 0 ]; then
        echo "Failed building $bm"
        echo "For more information see $BUILD_LOG_FILE"
        exit $ret
    fi
    touch ${TEST_DIR}/$bm/BUILD_SUCCESS
done

echo "Start testing all benchmarks"

for bm in ${BENCHMARKS[@]}; do
    echo "Testing $bm..."
    if [ -f  ${TEST_DIR}/$bm/TEST_SUCCESS ]; then
        continue
    fi
    cd $TEST_DIR
    ret=0
    cd $bm
    if [ $bm = "b_eff" ]; then
        cd bin
        rm kernel_*put_ch*
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
    if [ $bm = "PTRANS" ]; then
        cd bin
        rm kernel_*put_ch*
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
    make XCL_EMULATION_MODE=sw_emu CTEST_OUTPUT_ON_FAILURE=1 test &>> $TEST_LOG_FILE
    ret=$(($ret + $?))
    if [ $ret -ne 0 ]; then
        echo "Failed testing $bm"
        echo "For more information see $TEST_LOG_FILE"
        exit $ret
    fi
    touch ${TEST_DIR}/$bm/TEST_SUCCESS
done

echo "-----------"
echo "SUCCESS!!!!"
echo "-----------"
