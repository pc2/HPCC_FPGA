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
#ifndef SRC_HOST_COMMON_BENCHMARK_IO_H_
#define SRC_HOST_COMMON_BENCHMARK_IO_H_


/* Project's headers */
#include "program_settings.h"

/* External library headers */
#include "CL/cl.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#define ENTRY_SPACE 15

/**
* Parses and returns program options using the cxxopts library.
* The parsed parameters are depending on the benchmark that is implementing
* function.
* The header file is used to specify a unified interface so it can also be used
* in the testing binary.
* cxxopts is used to parse the parameters.
* @see https://github.com/jarro2783/cxxopts
* 
* @param argc Number of input parameters as it is provided by the main function
* @param argv Strings containing the input parameters as provided by the main function
* 
* @return program settings that are created from the given program arguments
*/
std::shared_ptr<ProgramSettings>
parseProgramParameters(int argc, char *argv[]);


/**
 * Prints the used configuration to std out before starting the actual benchmark.
 *
 * @param programSettings The program settings that are parsed from the command line
                            using parseProgramParameters
 * @param device The device selected for execution
 */
void 
printFinalConfiguration(const std::shared_ptr<ProgramSettings> &programSettings,
                        const cl::Device &device);

#endif
