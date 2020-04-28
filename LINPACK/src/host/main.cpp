//
// Created by Marius Meyer on 04.12.19.
//

#include "linpack_functionality.hpp"
#include "setup/fpga_setup.hpp"
#include "execution.h"

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Setup benchmark
    std::shared_ptr<ProgramSettings> programSettings =
            parseProgramParameters(argc, argv);
    fpga_setup::setupEnvironmentAndClocks();
    std::vector<cl::Device> usedDevice =
            fpga_setup::selectFPGADevice(programSettings->defaultPlatform,
                                         programSettings->defaultDevice);
    cl::Context context = cl::Context(usedDevice);
    cl::Program program = fpga_setup::fpgaSetup(&context, usedDevice,
                                                &programSettings->kernelFileName);

    printFinalConfiguration(programSettings, usedDevice[0]);

    std::shared_ptr<bm_execution::ExecutionConfiguration> config(
            new bm_execution::ExecutionConfiguration {
                    context, usedDevice[0], program,
                    programSettings->numRepetitions,
                    programSettings->matrixSize
            });

    HOST_DATA_TYPE *A, *b;
    cl_int* ipvt;

    posix_memalign(reinterpret_cast<void**>(&A), 4096, programSettings->matrixSize *
                                                                 programSettings->matrixSize * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&b), 4096, programSettings->matrixSize * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&ipvt), 4096, programSettings->matrixSize * sizeof(cl_int));


    HOST_DATA_TYPE norma;
    generateInputData(A, b, ipvt, programSettings->matrixSize, &norma);

    auto timing =  bm_execution::calculate(config, A, b, ipvt);

    double error = checkLINPACKresults(b, programSettings->matrixSize);

    free(A);
    free(b);
    free(ipvt);

    printResults(timing, programSettings->matrixSize);

    return error < 100 ? 0 : 1;
}

