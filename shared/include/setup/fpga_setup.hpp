/*
Copyright (c) 2019 Marius Meyer

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
#ifndef SRC_HOST_FPGA_SETUP_H_
#define SRC_HOST_FPGA_SETUP_H_

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <memory>

/* External libraries */
#ifdef USE_DEPRECATED_HPP_HEADER
#include "CL/cl.hpp"
#else
#include OPENCL_HPP_HEADER
#endif


/**
Makro to convert the error integer representation to its string representation
Source: https://gist.github.com/allanmac/9328bb2d6a99b86883195f8f78fd1b93
*/
#define CL_ERR_TO_STR(err) case err: return #err

namespace fpga_setup {

/**
 * @brief Base expection for all exception that
 *          are thrown by the fpga setup functions
 * 
 */
class FpgaSetupException : public std::exception
{
    public:

    /**
     * @brief Construct a new Fpga Setup Exception object
     * 
     * @param message The message that is printed by the what() function
     */
    FpgaSetupException(std::string message);

    /**
     * @brief Return the contained error message
     * 
     * @return const char* the error message as C string
     */
    const char*
    what() const noexcept override;

    private:
    std::string error_message;
};

/**
 * @brief Exception that is thrown if the ASSERT_CL failed
 * 
 */
class OpenClException : public FpgaSetupException
{
    public:

    /**
     * @brief Construct a new Open Cl Exception object
     * 
     * @param error_name The string representation of the 
     *                      OpenCL error
     */
    OpenClException(std::string  error_name);
};

    /**
Converts the reveived OpenCL error to a string

@param err The OpenCL error code

@return The string representation of the OpenCL error code
*/
    std::string
    getCLErrorString(cl_int const err);

/**
* Check the OpenCL return code for errors.
* If an error is detected, it will be printed and the programm execution is
* stopped.
*
* @param err The OpenCL error code
* @param file Name of the file that is checking the CL error
* @param line Line number in the file that is checking the CL error
*
* @throw OpenClException if the return code differs from CL_SUCCESS
*/
    void
    handleClReturnCode(cl_int const err, std::string const file,
                       int const line);

/**
Makro that enables checks for OpenCL errors with handling of the file and
line number.
*/
#define ASSERT_CL(err) fpga_setup::handleClReturnCode(err, __FILE__, __LINE__);

/**
Sets up the given FPGA with the kernel in the provided file.

@param context The context used for the program
@param program The devices used for the program
@param usedKernelFile The path to the kernel file
@return The program that is used to create the benchmark kernels
*/
    std::unique_ptr<cl::Program>
    fpgaSetup(const cl::Context *context, std::vector<cl::Device> deviceList,
              const std::string *usedKernelFile);

/**
Sets up the C++ environment by configuring std::cout and checking the clock
granularity using bm_helper::checktick()
*/
    void
    setupEnvironmentAndClocks();


/**
Searches an selects an FPGA device using the CL library functions.
If multiple platforms or devices are given, the user will be prompted to
choose a device.

@param defaultPlatform The index of the platform that has to be used. If a
                        value < 0 is given, the platform can be chosen
                        interactively
@param defaultDevice The index of the device that has to be used. If a
                        value < 0 is given, the device can be chosen
                        interactively

@return A list containing a single selected device
*/
    std::unique_ptr<cl::Device>
    selectFPGADevice(int defaultPlatform, int defaultDevice);

}  // namespace fpga_setup
#endif  // SRC_HOST_FPGA_SETUP_H_
