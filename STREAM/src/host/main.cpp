//
// Created by Marius Meyer on 04.12.19.
//

#include "stream_functionality.hpp"
#include "program_settings.h"
#include "setup/common_benchmark_io.hpp"
#include "CL/opencl.h"

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
                    programSettings->kernelReplications,
                    programSettings->streamArraySize,
                    programSettings->useMemoryInterleaving,
                    programSettings->useSingleKernel
            });

    HOST_DATA_TYPE *A, *B, *C;
#ifdef INTEL_FPGA
#ifdef USE_SVM
    A = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            programSettings->streamArraySize * sizeof(HOST_DATA_TYPE), 1024));
    B = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            programSettings->streamArraySize * sizeof(HOST_DATA_TYPE), 1024));
    C = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            programSettings->streamArraySize * sizeof(HOST_DATA_TYPE), 1024));
#else
    posix_memalign(reinterpret_cast<void**>(&A), 64, programSettings->streamArraySize * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&B), 64, programSettings->streamArraySize * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C), 64, programSettings->streamArraySize * sizeof(HOST_DATA_TYPE));
#endif
#endif
#ifdef XILINX_FPGA
    posix_memalign(reinterpret_cast<void**>(&A), 4096, programSettings->streamArraySize * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&B), 4096, programSettings->streamArraySize * sizeof(HOST_DATA_TYPE));
    posix_memalign(reinterpret_cast<void**>(&C), 4096, programSettings->streamArraySize * sizeof(HOST_DATA_TYPE));
#endif
    generateInputData(A, B, C, programSettings->streamArraySize);

    auto timing =  bm_execution::calculate(config, A, B, C);

    double error = checkSTREAMResult(A, B, C, programSettings->numRepetitions, programSettings->streamArraySize);

#ifdef USE_SVM
    clSVMFree(context(), reinterpret_cast<void *>(A));
    clSVMFree(context(), reinterpret_cast<void *>(B));
    clSVMFree(context(), reinterpret_cast<void *>(C));
#else
    free(A);
    free(B);
    free(C);
#endif

    printResults(timing);

    return error < 1 ? 0 : 1;
}

