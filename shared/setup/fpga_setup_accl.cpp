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

#ifdef _USE_MPI_
#include "mpi.h"
#endif

namespace fpga_setup {

    std::unique_ptr<ACCL::ACCL>
    fpgaSetupACCL(xrt::device &context,
              xrt::uuid &program) {
        int current_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, & current_rank);

        int current_size;
        MPI_Comm_size(MPI_COMM_WORLD, & current_size);

	std::vector<ACCL::rank_t> ranks = {};
        for (int i = 0; i < current_size; ++i) {
		ACCL::rank_t new_rank = {"127.0.0.1", 5500 + i, i,
                       1024};
            ranks.emplace_back(new_rank);
        }
	// TODO: Add start port here. Currenty hardcoded!
        return std::unique_ptr<ACCL::ACCL>(new ACCL::ACCL(ranks, current_rank,
                          "tcp://localhost:" +
                            std::to_string(5500 + current_rank)));
    }

}  // namespace fpga_setup
