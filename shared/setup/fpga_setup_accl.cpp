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

void configure_tcp(ACCL::BaseBuffer &tx_buf_network, ACCL::BaseBuffer &rx_buf_network,
                   xrt::kernel &network_krnl, std::vector<ACCL::rank_t> &ranks,
                   int rank) {
  std::cout << "Configure TCP Network Kernel" << std::endl;
  tx_buf_network.sync_to_device();
  rx_buf_network.sync_to_device();

  uint local_fpga_ip = ACCL::ip_encode(ranks[rank].ip);
  std::cout << "rank: " << rank << " FPGA IP: " << std::hex << local_fpga_ip
            << std::endl;

  network_krnl(local_fpga_ip, static_cast<uint32_t>(rank), local_fpga_ip,
               *(tx_buf_network.bo()), *(rx_buf_network.bo()));
}

ACCLContext fpgaSetupACCL(xrt::device &device, xrt::uuid &program,
                                          hpcc_base::BaseSettings &programSettings) {
  int current_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

  int current_size;
  MPI_Comm_size(MPI_COMM_WORLD, &current_size);

  std::vector<ACCL::rank_t> ranks = {};
  for (int i = 0; i < current_size; ++i) {
    // TODO: Replace the ip addresses and ports here for execution of real hardware?
    ACCL::rank_t new_rank = {"10.10.10." + std::to_string(i), 6000 + i, i, programSettings.acclBufferSize};
    ranks.emplace_back(new_rank);
  }

  ACCLContext accl;

  if (!programSettings.useAcclEmulation) {
    std::cout << "Create cclo ip" << std::endl;
    auto cclo_ip = xrt::ip(device, program, "ccl_offload:{ccl_offload_" + std::to_string(0) + "}");
    std::cout << "Create hostctrl" << std::endl;
    auto hostctrl_ip = xrt::kernel(device, program, "hostctrl:{hostctrl_" + std::to_string(0) + "}",
                                   xrt::kernel::cu_access_mode::exclusive);
    if (programSettings.acclProtocol == ACCL::networkProtocol::UDP) {
      std::cout << "Create CMAC" << std::endl;
      auto cmac = CMAC(xrt::ip(device, program, "cmac_0:{cmac_0}"));
      std::cout << "Create Network Layer" << std::endl;
      auto network_layer = Networklayer(
            xrt::ip(device, program, "networklayer:{networklayer_0}"));
      std::cout << "Configure VNX" << std::endl;
      configure_vnx(cmac, network_layer, ranks, current_rank);
    }
    if (programSettings.acclProtocol == ACCL::networkProtocol::TCP) {
      auto network_krnl = xrt::kernel(device, program, "network_krnl:{network_krnl_0}",
                      xrt::kernel::cu_access_mode::exclusive);
      accl.tx_buf_network = std::unique_ptr<ACCL::BaseBuffer>(new ACCL::FPGABuffer<int8_t>(
          64 * 1024 * 1024, ACCL::dataType::int8, device, network_krnl.group_id(3)));
      accl.rx_buf_network = std::unique_ptr<ACCL::BaseBuffer>(new ACCL::FPGABuffer<int8_t>(
          64 * 1024 * 1024, ACCL::dataType::int8, device, network_krnl.group_id(4)));
      configure_tcp(*accl.tx_buf_network, *accl.rx_buf_network, network_krnl, ranks, current_rank);
    }
    std::vector<int> mem(1, 0);
    std::cout << "Create ACCL" << std::endl;
    accl.accl = std::unique_ptr<ACCL::ACCL>(
        new ACCL::ACCL(ranks, current_rank, device, cclo_ip, hostctrl_ip, 0, mem, programSettings.acclProtocol, programSettings.acclBufferCount, programSettings.acclBufferSize));
  } else {
    // TODO: Add start port here. Currenty hardcoded!
    accl.accl = std::unique_ptr<ACCL::ACCL>(
        new ACCL::ACCL(ranks, current_rank, 6000, device, programSettings.acclProtocol, programSettings.acclBufferCount, programSettings.acclBufferSize));
  }

  if (programSettings.acclProtocol == ACCL::networkProtocol::TCP) {
    MPI_Barrier(MPI_COMM_WORLD);
    accl.accl->open_port();
    MPI_Barrier(MPI_COMM_WORLD);
    accl.accl->open_con();
  }
  return accl;
}

} // namespace fpga_setup
