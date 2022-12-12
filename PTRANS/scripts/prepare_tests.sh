#!/usr/bin/bash

cd $1
touch kernel_output_ch0
touch kernel_output_ch1
touch kernel_output_ch2
touch kernel_output_ch3
ln -s kernel_output_ch0 kernel_input_ch1
ln -s kernel_output_ch2 kernel_input_ch3
ln -s kernel_output_ch1 kernel_input_ch0
ln -s kernel_output_ch3 kernel_input_ch2
