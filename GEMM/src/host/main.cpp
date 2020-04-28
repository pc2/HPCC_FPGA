//
// Created by Marius Meyer on 04.12.19.
//

#include "parameters.h"
#include "gemm_functionality.hpp"
#include "setup/fpga_setup.hpp"

/**
The program entry point
*/
int main(int argc, char * argv[]) {
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

    // Give setup summary
    std::cout << "Summary:" << std::endl
              << "Kernel Repetitions:  " << programSettings->numRepetitions
              << std::endl
              << "Total matrix size:   " << programSettings->matrixSize
              << std::endl
              << "Memory Interleaving: " << programSettings->useMemInterleaving
              << std::endl
              << "Kernel file:         " << programSettings->kernelFileName
              << std::endl
              << "Device:              "
              << usedDevice[0].getInfo<CL_DEVICE_NAME>() << std::endl
              << "Verification:        "
              #ifdef _USE_BLAS_
              << "external library"
              #else
              << "internal ref. implementation"
              #endif
              << std::endl 
              << HLINE
              << "Start benchmark using the given configuration." << std::endl
              << HLINE;

    std::shared_ptr<bm_execution::ExecutionConfiguration> config(
            new bm_execution::ExecutionConfiguration{
                    context, usedDevice[0], program,
                    programSettings->kernelName,
                    programSettings->numRepetitions,
                    programSettings->matrixSize,
                    programSettings->useMemInterleaving
            });

    uint lda = config->matrixSize;
    HOST_DATA_TYPE* a;
    posix_memalign(reinterpret_cast<void**>(&a), 64,
                   sizeof(HOST_DATA_TYPE)*lda*config->matrixSize);
    HOST_DATA_TYPE* b;
    posix_memalign(reinterpret_cast<void**>(&b), 64,
                   sizeof(HOST_DATA_TYPE)*lda*config->matrixSize);
    HOST_DATA_TYPE* c;
    posix_memalign(reinterpret_cast<void**>(&c), 64,
                   sizeof(HOST_DATA_TYPE)*lda*config->matrixSize);
    HOST_DATA_TYPE* c_out;
    posix_memalign(reinterpret_cast<void**>(&c_out), 64,
                   sizeof(HOST_DATA_TYPE)*lda*config->matrixSize);

    HOST_DATA_TYPE alpha = 0.5;
    HOST_DATA_TYPE beta = 2.0;

    HOST_DATA_TYPE norma;
    matgen(a, 1, config->matrixSize, config->matrixSize, &norma);
    matgen(b, 2, config->matrixSize, config->matrixSize, &norma);
    matgen(c, 3, config->matrixSize, config->matrixSize, &norma);

    // Start actual benchmark
    auto results = bm_execution::calculate(config, a, b, c , c_out, alpha, beta);

    /* --- Check Results --- */

    double error = checkGEMMresults(c_out, config->matrixSize, config->matrixSize);

    free(reinterpret_cast<void *>(a));
    free(reinterpret_cast<void *>(b));
    free(reinterpret_cast<void *>(c));

    printResults(results, programSettings->matrixSize);

    return error > 0.1 ? 1 : 0;
}

