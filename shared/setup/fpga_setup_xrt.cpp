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

std::unique_ptr<xrt::device> selectFPGADevice(int defaultDevice)
{
    int current_device;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_device);
    if (defaultDevice >= 0) {
        current_device = defaultDevice;
    } else {
        // TODO Use xrt::system::enumerate_devices() in "experimental/xrt_system.h" for future XRT versions
        //  instead of hardcoded number of devices.
        current_device = current_device % 3;
    }
    return std::unique_ptr<xrt::device>(new xrt::device(current_device));
}
} // namespace fpga_setup
