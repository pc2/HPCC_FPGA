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
#ifndef SRC_HOST_FPGA_SETUP_UDP_H_
#define SRC_HOST_FPGA_SETUP_UDP_H_

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* External libraries */
#include "hpcc_settings.hpp"
#include "xrt/xrt_device.h"
#include <vnx/cmac.hpp>
#include <vnx/networklayer.hpp>

namespace fpga_setup
{

struct VNXContext {
    std::vector<vnx::Networklayer> udps;
    std::vector<vnx::CMAC> cmacs;
};

/**
Sets up the given FPGA with the kernel in the provided file.

@param device The device used for the program
@param program The program used to find the ACCL kernels for hardware execution
@param programSettings Pass current program settings to configure ACCL according to user specification
*/
std::unique_ptr<VNXContext> fpgaSetupUDP(xrt::device &device, xrt::uuid &program,
                                         hpcc_base::BaseSettings &programSettings);

} // namespace fpga_setup
#endif // SRC_HOST_FPGA_SETUP_H_
