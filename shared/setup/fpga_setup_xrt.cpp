//
// Created by Marius Meyer on 04.12.19.
//

#include "setup/fpga_setup_xrt.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/* External libraries */
#include "parameters.h"

#include "xrt/xrt_device.h"
#ifdef _USE_MPI_
#include "mpi.h"
#endif

namespace fpga_setup
{

std::unique_ptr<xrt::uuid> fpgaSetup(xrt::device &device, const std::string &kernelFileName)
{
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    int current_size;
    MPI_Comm_size(MPI_COMM_WORLD, &current_size);

    return std::unique_ptr<xrt::uuid>(new xrt::uuid(device.load_xclbin(kernelFileName)));
}

std::string bdf_map(uint32_t device_id)
{
    if (device_id == 0) {
        return "0000:a1:00.1";
    } else if (device_id == 1) {
        return "0000:81:00.1";
    } else if (device_id == 2) {
        return "0000:01:00.1";
    } else {
        throw std::invalid_argument("Invalid device id");
    }
}

std::unique_ptr<xrt::device> selectFPGADevice(int defaultDevice)
{
    int current_device;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_device);
    if (defaultDevice < 0) {
        defaultDevice = 0;
    }
    // use user passed defaultDevice as offset for device selection
    // default to zero offset
    current_device = (current_device + defaultDevice) % 3;
    return std::unique_ptr<xrt::device>(new xrt::device(bdf_map(current_device)));
}
} // namespace fpga_setup
