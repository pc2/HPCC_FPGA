//
// Created by Marius Meyer on 04.12.19.
//

#include "setup/fpga_setup_accl.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

/* External libraries */
#include "parameters.h"
#include "xrt.h"
#ifdef _USE_MPI_
#include "mpi.h"
#endif

namespace fpga_setup {

    std::unique_ptr<ACCL::ACCL>
    fpgaSetupACCL(xrt::device &device,
              xrt::uuid &program) {
        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);

        int current_size;
        MPI_Comm_size(MPI_COMM_WORLD, & current_size);
 
        std::vector<ACCL::rank_t> ranks = {};
        for (int i = 0; i < current_size; ++i) {
		    // TODO: Replace the ip addresses and ports here for execution of real hardware?
            ACCL::rank_t new_rank = {"127.0.0.1", 5500 + i, i,
                       1024};
            ranks.emplace_back(new_rank);
        }
#ifdef ACCL_HARDWARE_SUPPORT
        auto cclo_ip = xrt::ip(device, program, "ccl_offload:{ccl_offload_" + std::to_string(0) + "}");
        auto hostctl_ip = xrt::kernel(device, program, "hostctrl:{hostctrl_" + std::to_string(0) + "}",
                xrt::kernel::cu_access_mode::exclusive);
        return std::unique_ptr<ACCL::ACCL>(new ACCL::ACCL(ranks, rank, device, cclo_ip, hostctrl_ip, 0, {0}, 0);
#else
                // TODO: Add start port here. Currenty hardcoded!
        return std::unique_ptr<ACCL::ACCL>(new ACCL::ACCL(ranks, current_rank,
                          "tcp://localhost:" +
                            std::to_string(5500 + current_rank)));
#endif
        }

}  // namespace fpga_setup
