//
// Created by Marius Meyer on 04.12.19.
//

#include "random_access_functionality.hpp"
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
                    programSettings->numReplications,
                    programSettings->dataSize
            });

    HOST_DATA_TYPE *data;
#ifdef USE_SVM
    data = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            programSettings->dataSize * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&data), 4096, programSettings->dataSize * sizeof(HOST_DATA_TYPE));
#endif

    generateInputData(data, programSettings->dataSize);

    auto timing =  bm_execution::calculate(config, data);

    double error = checkRandomAccessResults(data, programSettings->dataSize);

#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void *>(data));
#else
    free(data);
#endif

    printResults(timing, programSettings->dataSize, error);

    return error < 0.01 ? 0 : 1;
}

