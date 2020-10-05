#!/bin/bash
# Author: Marius Meyer
#
# This file can be used as awrapper to clean the files that are used as bufer for the external channels during emulation
# They might contain old data from previous runs and might thus affect the benchmark result
# The script emptys the files and then calls the actual application that is given as parameters with the script.
#
# e.g. ./clean_emulation_output_files [Folder that contains the channel files] ./Benchmark -param1 -param2
#
#

echo "" > $1/kernel_output_ch0
echo "" > $1/kernel_output_ch1
echo "" > $1/kernel_output_ch2
echo "" > $1/kernel_output_ch3

${@:2}
