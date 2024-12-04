//
// Created by Marius Meyer on 04.12.19.
//

#include "setup/fpga_setup_udp.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

/* External libraries */
#include "experimental/xrt_ip.h"
#include "parameters.h"
#include "xrt/xrt_kernel.h"
#include <vnx/cmac.hpp>
#include <vnx/networklayer.hpp>
#ifdef _USE_MPI_
#include "mpi.h"
#endif

using namespace vnx;

namespace fpga_setup
{

void configure_vnx(CMAC &cmac, Networklayer &network_layer, int ranks, int rank, int offset)
{
    cmac.link_status();
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

    network_layer.update_ip_address("10.10.10." + std::to_string(offset + rank));
    for (size_t i = 0; i < ranks; ++i) {
        if (i == static_cast<size_t>(offset + rank)) {
            continue;
        }
        int index = i;
        if (i > static_cast<size_t>(offset + rank)) {
            index--;
        }
        network_layer.configure_socket(index, "10.10.10." + std::to_string(i), 5000, 5000, true);
    }

    network_layer.populate_socket_table();

    std::this_thread::sleep_for(std::chrono::seconds(4));
    network_layer.arp_discovery();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    network_layer.arp_discovery();
}

std::unique_ptr<VNXContext> fpgaSetupUDP(xrt::device &device, xrt::uuid &program,
                                         hpcc_base::BaseSettings &programSettings)
{
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    int current_size;
    MPI_Comm_size(MPI_COMM_WORLD, &current_size);

    auto context = std::unique_ptr<VNXContext>(new VNXContext());

    std::vector<std::thread> conf_threads;
    for (int i = 0; i < programSettings.kernelReplications; i++) {
        auto cmac = CMAC(xrt::ip(device, program, "cmac_" + std::to_string(i) + ":{cmac_" + std::to_string(i) + "}"));
        auto network_layer =
            Networklayer(xrt::ip(device, program, "networklayer:{networklayer_" + std::to_string(i) + "}"));
        conf_threads.emplace_back(configure_vnx, std::ref(cmac), std::ref(network_layer), current_size * programSettings.kernelReplications, current_rank,
                      i * current_size);
        auto arp_tab = network_layer.read_arp_table(current_size * programSettings.kernelReplications);
        for (auto const &e : arp_tab) {
            std::cout << e.second.first << " -> " << e.second.second << std::endl;
        }
        context->udps.push_back(network_layer);
        context->cmacs.push_back(cmac);
    }

    for (auto & t: conf_threads) {
        t.join();
    }
    return context;
}

} // namespace fpga_setup
