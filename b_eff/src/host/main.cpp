//
// Created by Marius Meyer on 04.12.19.
//

#include <memory>

#include "network_functionality.hpp"
#include "mpi.h"

/**
The program entry point
*/
int
main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Setup benchmark
    std::shared_ptr<ProgramSettings> programSettings =
            parseProgramParameters(argc, argv);
    if (world_rank == 0) {
        fpga_setup::setupEnvironmentAndClocks();
    }
    std::vector<cl::Device> usedDevice =
            fpga_setup::selectFPGADevice(programSettings->defaultPlatform,
                                         programSettings->defaultDevice);
    cl::Context context = cl::Context(usedDevice);
    cl::Program program = fpga_setup::fpgaSetup(&context, usedDevice,
                                                &programSettings->kernelFileName);

    if (world_rank == 0) {
        printFinalConfiguration(programSettings, usedDevice[0]);
    }

    std::shared_ptr<bm_execution::ExecutionConfiguration> config(
            new bm_execution::ExecutionConfiguration {
                    context, usedDevice[0], program,
                    programSettings->numRepetitions
            });

    auto msgSizes = getMessageSizes();
    std::vector<std::shared_ptr<bm_execution::ExecutionTimings>> timing_results;

    for (cl_uint size : msgSizes) {
        if (world_rank == 0) {
            std::cout << "Measure for " << size << " Byte" << std::endl;
        }
        cl_uint looplength = std::max((programSettings->looplength) / ((size + (CHANNEL_WIDTH)) / (CHANNEL_WIDTH)), 1u);
        timing_results.push_back(bm_execution::calculate(config, size,looplength));
    }

    // Collect the measurement results from all other nodes
    bm_execution::CollectedResultMap collected_results;
    if (world_rank > 0) {
        for (const auto& t : timing_results) {
            MPI_Send(&(t->messageSize),
                     1,
                     MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&(t->looplength),
                     1,
                     MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&(t->calculationTimings.front()),
                     programSettings->numRepetitions,
                     MPI_DOUBLE, 0,  2, MPI_COMM_WORLD);
        }
    } else {
        std::cout << "Collect results over MPI.";
        int k = 0;
        for (int size : msgSizes) {
            std::vector<std::shared_ptr<bm_execution::ExecutionTimings>> tmp_timings;
            std::cout << ".";
            for (int i=1; i < world_size; i++) {
                auto execution_result = std::make_shared<bm_execution::ExecutionTimings>( bm_execution::ExecutionTimings {
                    0,0,std::vector<double>(programSettings->numRepetitions)
                });
                MPI_Status status;
                MPI_Recv(&(execution_result->messageSize),
                         1,
                         MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&(execution_result->looplength),
                         1,
                         MPI_UNSIGNED, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&(execution_result->calculationTimings.front()),
                         programSettings->numRepetitions,
                         MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
                tmp_timings.push_back(execution_result);
                if (execution_result->messageSize != size) {
                    std::cerr << "Wrong message size: " << execution_result->messageSize << " != " << size << " from rank " << i << std::endl;
                    exit(2);
                }
            }
            tmp_timings.push_back(timing_results[k]);
            k++;
            collected_results.emplace(size, std::make_shared<std::vector<std::shared_ptr<bm_execution::ExecutionTimings>>>(tmp_timings));
        }
        std::cout << " done!" << std::endl;
    }

    if (world_rank == 0) {
        printResults(collected_results);
    }

    MPI_Finalize();
    return 0;
}

