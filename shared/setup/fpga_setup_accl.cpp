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
#include <vnx/cmac.hpp>
#include <vnx/networklayer.hpp>
#include "xrt/xrt_kernel.h"
#ifdef _USE_MPI_
#include "mpi.h"
#endif

using namespace vnx;

namespace fpga_setup {

void configure_vnx(CMAC &cmac, Networklayer &network_layer,
                   std::vector<ACCL::rank_t> &ranks, int rank) {
  if (ranks.size() > max_sockets_size) {
    throw std::runtime_error("Too many ranks. VNX supports up to " +
                             std::to_string(max_sockets_size) + " sockets.");
  }

  const auto link_status = cmac.link_status();

  if (link_status.at("rx_status")) {
    std::cout << "Link successful!" << std::endl;
  } else {
    std::cout << "No link found." << std::endl;
  }

  if (!link_status.at("rx_status")) {
    // Give time for other ranks to setup link.
    std::this_thread::sleep_for(std::chrono::seconds(3));
    exit(1);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  network_layer.update_ip_address(ranks[rank].ip);
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (i == static_cast<size_t>(rank)) {
      continue;
    }

    network_layer.configure_socket(i, ranks[i].ip, ranks[i].port,
                                   ranks[rank].port, true);
  }

  network_layer.populate_socket_table();

  std::this_thread::sleep_for(std::chrono::seconds(4));
  network_layer.arp_discovery();
  std::this_thread::sleep_for(std::chrono::seconds(2));
  network_layer.arp_discovery();
}

std::unique_ptr<ACCL::ACCL> fpgaSetupACCL(xrt::device &device, xrt::uuid &program,
                                          bool useAcclEmulation) {
  int current_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

  int current_size;
  MPI_Comm_size(MPI_COMM_WORLD, &current_size);

  std::vector<ACCL::rank_t> ranks = {};
  for (int i = 0; i < current_size; ++i) {
    // TODO: Replace the ip addresses and ports here for execution of real hardware?
    ACCL::rank_t new_rank = {"10.10.10." + current_rank, 5500 + i, i, ACCL_BUFFER_SIZE};
    ranks.emplace_back(new_rank);
  }
  if (!useAcclEmulation) {
    std::cout << "Create cclo ip" << std::endl;
    auto cclo_ip = xrt::ip(device, program, "ccl_offload:{ccl_offload_" + std::to_string(0) + "}");
    std::cout << "Create hostctrl" << std::endl;
    auto hostctrl_ip = xrt::kernel(device, program, "hostctrl:{hostctrl_" + std::to_string(0) + "}",
                                   xrt::kernel::cu_access_mode::exclusive);
 
    auto cmac = CMAC(xrt::ip(device, program, "cmac_0:{cmac_0}"));
     auto network_layer = Networklayer(
          xrt::ip(device, program, "networklayer:{networklayer_0}"));
     configure_vnx(cmac, network_layer, ranks, current_rank);

    std::vector<int> mem(1, 0);
    std::cout << "Create ACCL" << std::endl;
    return std::unique_ptr<ACCL::ACCL>(
        new ACCL::ACCL(ranks, current_rank, device, cclo_ip, hostctrl_ip, 0, mem, 0, ACCL::networkProtocol::UDP));
  } else {
    // TODO: Add start port here. Currenty hardcoded!
    return std::unique_ptr<ACCL::ACCL>(
        new ACCL::ACCL(ranks, current_rank, 5500, device, ACCL::networkProtocol::UDP, 16, ACCL_BUFFER_SIZE));
  }
}

} // namespace fpga_setup
