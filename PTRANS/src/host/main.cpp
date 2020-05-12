//
// Created by Marius Meyer on 04.12.19.
//

#include "transpose_functionality.hpp"

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Setup benchmark
    std::shared_ptr<ProgramSettings> programSettings =
            parseProgramParameters(argc, argv);
    fpga_setup::setupEnvironmentAndClocks();

    std::vector<cl::Device> usedDevice;
    cl::Context context;
    cl::Program program;
    cl::Device device;

    if (programSettings->kernelFileName != "CPU") {
        usedDevice = fpga_setup::selectFPGADevice(
                programSettings->defaultPlatform,
                programSettings->defaultDevice);
        context = cl::Context(usedDevice);
        program = fpga_setup::fpgaSetup(&context, usedDevice, &programSettings->kernelFileName);
    }

    if (usedDevice.size() > 0) {
        device = usedDevice[0];
        printFinalConfiguration(programSettings, device);
    }

    std::shared_ptr<bm_execution::ExecutionConfiguration> config(
            new bm_execution::ExecutionConfiguration{
                    context, device, program,
                    programSettings->kernelName,
                    programSettings->numRepetitions,
                    programSettings->matrixSize * programSettings->blockSize,
                    programSettings->blockSize,
                    programSettings->useMemInterleaving
            });

    HOST_DATA_TYPE* A;
    HOST_DATA_TYPE* B;
    HOST_DATA_TYPE* result;

    posix_memalign(reinterpret_cast<void **>(&A), 64,
                   sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize);
    posix_memalign(reinterpret_cast<void **>(&B), 64,
                   sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize);
    posix_memalign(reinterpret_cast<void **>(&result), 64,
                   sizeof(HOST_DATA_TYPE) * config->matrixSize * config->matrixSize);

    generateInputData(config->matrixSize, A, B);
    // Start actual benchmark
    auto results = bm_execution::calculate(config, A, B, result);

    printResults(results, config->matrixSize);

    double error =  printCalculationError(config->matrixSize, result);


    return (error < 1.0e-5) ? 0 : 1;
}

