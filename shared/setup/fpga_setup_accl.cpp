//
// Created by Marius Meyer on 04.12.19.
//

#include "setup/fpga_setup_accl.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/* External libraries */
#include "experimental/xrt_ip.h"
#include "parameters.h"
#include "xrt/xrt_kernel.h"
#ifdef _USE_MPI_
#include "mpi.h"
#endif

namespace fpga_setup {

std::unique_ptr<ACCL::ACCL> fpgaSetupACCL(xrt::device &device, xrt::uuid &program,
                                          bool useAcclEmulation) {
  int current_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

  int current_size;
  MPI_Comm_size(MPI_COMM_WORLD, &current_size);

  std::vector<ACCL::rank_t> ranks = {};
  for (int i = 0; i < current_size; ++i) {
    // TODO: Replace the ip addresses and ports here for execution of real hardware?
    ACCL::rank_t new_rank = {"127.0.0.1", 5500 + i, i, 1024};
    ranks.emplace_back(new_rank);
  }
  if (!useAcclEmulation) {
    auto cclo_ip = xrt::ip(device, program, "ccl_offload:{ccl_offload_" + std::to_string(0) + "}");
    auto hostctrl_ip = xrt::kernel(device, program, "hostctrl:{hostctrl_" + std::to_string(0) + "}",
                                   xrt::kernel::cu_access_mode::exclusive);
    std::vector<int> mem(1, 0);
    return std::unique_ptr<ACCL::ACCL>(
        new ACCL::ACCL(ranks, current_rank, device, cclo_ip, hostctrl_ip, 0, mem, 0));
  } else {
    // TODO: Add start port here. Currenty hardcoded!
    return std::unique_ptr<ACCL::ACCL>(
        new ACCL::ACCL(ranks, current_rank, 5500, device, ACCL::networkProtocol::TCP, 16, 1024));
  }
}

} // namespace fpga_setup
