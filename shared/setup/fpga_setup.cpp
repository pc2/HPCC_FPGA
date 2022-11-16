//
// Created by Marius Meyer on 04.12.19.
//

#include "setup/fpga_setup.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

/* External libraries */
#include "parameters.h"

#ifdef _USE_MPI_
#include "mpi.h"
#endif

namespace fpga_setup {

FpgaSetupException::FpgaSetupException(std::string message) : error_message(message) {}

const char*
FpgaSetupException::what() const noexcept
{
    return error_message.c_str();
}

OpenClException::OpenClException(std::string error_name)
    : FpgaSetupException("An OpenCL error occured: " + error_name) {}

/**
Converts the reveived OpenCL error to a string

@param err The OpenCL error code

@return The string representation of the OpenCL error code
*/
    std::string
    getCLErrorString(cl_int const err) {
        switch (err) {
            CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
            CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
            CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
            CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
            CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
            CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
            CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
            CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
            CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
            CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
            CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
            CL_ERR_TO_STR(CL_MAP_FAILURE);
            CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
            CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
            CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
            CL_ERR_TO_STR(CL_INVALID_VALUE);
            CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
            CL_ERR_TO_STR(CL_INVALID_PLATFORM);
            CL_ERR_TO_STR(CL_INVALID_DEVICE);
            CL_ERR_TO_STR(CL_INVALID_CONTEXT);
            CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
            CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
            CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
            CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
            CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
            CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
            CL_ERR_TO_STR(CL_INVALID_SAMPLER);
            CL_ERR_TO_STR(CL_INVALID_BINARY);
            CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
            CL_ERR_TO_STR(CL_INVALID_PROGRAM);
            CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
            CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
            CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
            CL_ERR_TO_STR(CL_INVALID_KERNEL);
            CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
            CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
            CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
            CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
            CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
            CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
            CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
            CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
            CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
            CL_ERR_TO_STR(CL_INVALID_EVENT);
            CL_ERR_TO_STR(CL_INVALID_OPERATION);
            CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
            CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
            CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
            CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
            CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
            CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
            CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
            CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
            CL_ERR_TO_STR(CL_INVALID_PROPERTY);
            CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
            CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
            CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
            CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);

            default:
                return "UNKNOWN ERROR CODE";
        }
    }


    void
    handleClReturnCode(cl_int const err, std::string const file,
                       int const line) {
        if (err != CL_SUCCESS) {
            std::string err_string = getCLErrorString(err);
            std::cerr << "ERROR in OpenCL library detected! Aborting."
                      << std::endl << file << ":" << line << ": " << err_string
                      << std::endl;
            throw fpga_setup::OpenClException(err_string);
        }
    }

/**
Sets up the given FPGA with the kernel in the provided file.

@param context The context used for the program
@param program The devices used for the program
@param usedKernelFile The path to the kernel file
@return The program that is used to create the benchmark kernels
*/
    std::unique_ptr<cl::Program>
    fpgaSetup(const cl::Context *context, std::vector<cl::Device> deviceList,
              const std::string *usedKernelFile) {
        int err;
        int world_rank = 0;

#ifdef _USE_MPI_
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif

        if (world_rank == 0) {
            std::cout << HLINE;
            std::cout << "FPGA Setup:" << usedKernelFile->c_str() << std::endl;
        }

        // Open file stream if possible
        std::ifstream aocxStream(usedKernelFile->c_str(), std::ifstream::binary);
        if (!aocxStream.is_open()) {
            std::cerr << "Not possible to open from given file!" << std::endl;
            throw FpgaSetupException("Not possible to open from given file: " + *usedKernelFile);
        }

        // Read in file contents and create program from binaries
        aocxStream.seekg(0, aocxStream.end);
        long file_size = aocxStream.tellg();
        aocxStream.seekg(0, aocxStream.beg);
        std::vector<unsigned char> buf(file_size);
        aocxStream.read(reinterpret_cast<char *>(buf.data()), file_size);


#ifdef USE_DEPRECATED_HPP_HEADER
        cl::Program::Binaries mybinaries;
        mybinaries.push_back({buf.data(), file_size});
#else
        cl::Program::Binaries mybinaries{buf};
#endif

        // Create the Program from the AOCX file.
        cl::Program program(*context, deviceList, mybinaries, NULL, &err);
        ASSERT_CL(err)

        // Build the program (required for fast emulation on Intel)
        ASSERT_CL(program.build());
        
        if (world_rank == 0) {
            std::cout << "Prepared FPGA successfully for global Execution!" <<
                      std::endl;
            std::cout << HLINE;
        }
        return std::unique_ptr<cl::Program>(new cl::Program(program));
    }

/**
Sets up the C++ environment by configuring std::cout and checking the clock
granularity using bm_helper::checktick()
*/
    void
    setupEnvironmentAndClocks() {
        std::cout << std::setprecision(5) << std::scientific;

        int world_rank = 0;

#ifdef _USE_MPI_
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif

        if (world_rank == 0) {
            std::cout << HLINE;
            std::cout << "General setup:" << std::endl;

            // Check clock granularity and output result
            std::cout << "C++ high resolution clock is used." << std::endl;
            std::cout << "The clock precision seems to be "
                    << static_cast<double>
                        (std::chrono::high_resolution_clock::period::num) /
                        std::chrono::high_resolution_clock::period::den * 10e9
                    << "ns" << std::endl;

            std::cout << HLINE;
        }
    }


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
@param platformString The platform string which should be chosen.
                        If it is empty, it will be ignored. If it is not empty,
                        but the string is not found an exception is thrown.

@return A list containing a single selected device
*/
    std::unique_ptr<cl::Device>
    selectFPGADevice(int defaultPlatform, int defaultDevice, std::string platformString) {
        // Integer used to store return codes of OpenCL library calls
        int err;

        int world_rank = 0;
        int world_size = 1;
        
#ifdef _USE_MPI_
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

        std::vector<cl::Platform> platformList;
        err = cl::Platform::get(&platformList);
        ASSERT_CL(err)

        // Choose the target platform
        long unsigned int chosenPlatformId = 0;
        if (defaultPlatform >= 0) {
            if (platformString.size() > 0) {
                bool found = false;
                for (int i = 0; i < platformList.size(); i++) {
                    if (platformList[i].getInfo<CL_PLATFORM_NAME>() == platformString) {
                        chosenPlatformId = i;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw FpgaSetupException("Invalid platform string specified: " + platformString);
                }
            } else if (defaultPlatform < static_cast<int>(platformList.size())) {
                chosenPlatformId = defaultPlatform;
            } else {
                std::cerr << "Default platform " << defaultPlatform
                          << " can not be used. Available platforms: "
                          << platformList.size() << std::endl;
                throw FpgaSetupException("Invalid platform index specified: " + std::to_string(defaultPlatform) + "/" + std::to_string(platformList.size() - 1));
            }
        } else if (platformList.size() > 1 && world_size == 1) {
            std::cout <<
                      "Multiple platforms have been found. Select the platform by"\
            " typing a number:" << std::endl;
            for (long unsigned int platformId = 0;
                 platformId < platformList.size(); platformId++) {
                std::cout << platformId << ") " <<
                          platformList[platformId].getInfo<CL_PLATFORM_NAME>() <<
                          std::endl;
            }
            std::cout << "Enter platform id [0-" << platformList.size() - 1
                      << "]:";
            std::cin >> chosenPlatformId;
        }
        cl::Platform platform = platformList[chosenPlatformId];
        if (world_rank == 0) {
            std::cout << "Selected Platform: "
                      << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        }
        std::vector<cl::Device> deviceList;
        err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &deviceList);
        ASSERT_CL(err)

        // Choose taget device
        long unsigned int chosenDeviceId = 0;
        if (defaultDevice >= 0) {
            if (defaultDevice < static_cast<int>(deviceList.size())) {
                chosenDeviceId = defaultDevice;
            } else {
                std::cerr << "Default device " << defaultDevice
                          << " can not be used. Available devices: "
                          << deviceList.size() << std::endl;
                throw FpgaSetupException("Invalid device index specified: " + std::to_string(defaultDevice) + "/" + std::to_string(deviceList.size() - 1));
            }
        } else if (deviceList.size() > 1) {
            if (world_size == 1) {
                    std::cout <<
                              "Multiple devices have been found. Select the device by"\
                            " typing a number:" << std::endl;

                for (long unsigned int deviceId = 0;
                     deviceId < deviceList.size(); deviceId++) {
                    std::cout << deviceId << ") " <<
                              deviceList[deviceId].getInfo<CL_DEVICE_NAME>() <<
                              std::endl;
                }
                std::cout << "Enter device id [0-" << deviceList.size() - 1 << "]:";
                std::cin >> chosenDeviceId;
            } else {
                chosenDeviceId = static_cast<long unsigned int>(world_rank % deviceList.size());
            }
        } else if (deviceList.size() == 1) {
            chosenDeviceId = 0;
        } else {
            throw std::runtime_error("No devices found for selected Platform!");
        }

        if (world_rank == 0) {
            // Give selection summary
            std::cout << HLINE;
            std::cout << "Selection summary:" << std::endl;
            std::cout << "Platform Name: " <<
                      platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::cout << "Device Name:   " <<
                      deviceList[chosenDeviceId].getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << HLINE;
        }

        return std::unique_ptr<cl::Device>(new cl::Device(deviceList[chosenDeviceId]));
    }

}  // namespace fpga_setup