/*
Copyright (c) 2022 Marius Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef SRC_HOST_FPGA_SETUP_XRT_H_
#define SRC_HOST_FPGA_SETUP_XRT_H_

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <memory>

/* External libraries */
#include "xrt/xrt_device.h"


namespace fpga_setup {

/**
Sets up the given FPGA with the kernel in the provided file.

@param device The device used for the program
@param usedKernelFile The path to the kernel file
@return The program that is used to create the benchmark kernels
*/
    std::unique_ptr<uuid>
    fpgaSetupXRT(xrt::device &device,
              const std::string *usedKernelFile);


/**
Searches an selects an FPGA device using the CL library functions.
If multiple platforms or devices are given, the user will be prompted to
choose a device.

@param defaultDevice The index of the device that has to be used. If a
                        value < 0 is given, the device can be chosen
                        interactively

@return the selected device
*/
    std::unique_ptr<xrt::device>
    selectFPGADeviceXRT(int defaultDevice);

}  // namespace fpga_setup
#endif  // SRC_HOST_FPGA_SETUP_H_
