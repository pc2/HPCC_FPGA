//
// Created by Marius Meyer on 04.12.19.
//

#include "mpi.h"
#include "network_benchmark.hpp"

using namespace network;

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Setup benchmark
    NetworkBenchmark bm(argc, argv);
    bool success = bm.executeBenchmark();
    MPI_Finalize();
    if (success) {
        return 0;
    }
    else {
        return 1;
    }
}


