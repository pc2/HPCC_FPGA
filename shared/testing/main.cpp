/*
Copyright (c) 2020 Marius Meyer

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

/* Project's headers */
#include "program_settings.h"
#include "setup/common_benchmark_io.hpp"
#include "setup/fpga_setup.hpp"
#include "test_program_settings.h"

#include "gtest/gtest.h"
#include "CL/cl.hpp"

std::shared_ptr<ProgramSettings> programSettings;

/**
The program entry point for the unit tests
*/
int
main(int argc, char *argv[]) {

    std::cout << "THIS BINARY EXECUTES UNIT TESTS FOR THE FOLLOWING BENCHMARK:" << std::endl << std::endl;

    ::testing::InitGoogleTest(&argc, argv);

    // Parse input parameters
    programSettings = parseProgramParameters(argc, argv);
    fpga_setup::setupEnvironmentAndClocks();

    std::vector<cl::Device> usedDevice =
    fpga_setup::selectFPGADevice(programSettings->defaultPlatform,
                                         programSettings->defaultDevice);

    // Print input parameters
    printFinalConfiguration(programSettings, usedDevice[0]);

    return RUN_ALL_TESTS();


}

