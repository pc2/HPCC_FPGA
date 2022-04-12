//
// Created by Marius Meyer on 04.12.19.
//

#include "setup/fpga_setup_xrt.hpp"

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

    std::unique_ptr<xrt::uuid>
    fpgaSetup(xrt::device &device,
              std::string &kernelFileName) {
        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);

        int current_size;
        MPI_Comm_size(MPI_COMM_WORLD, & current_size);

        return std::unique_ptr<xrt::uuid>(new device.load_xclbin(kernelFileName));
    }

    std::unique_ptr<xrt::device>
    selectFPGADevice(int defaultDevice) {
        return std::unique_ptr<xrt::device>(new xrt::device(defaultDevice));
    } 
}  // namespace fpga_setup
