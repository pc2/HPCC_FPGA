//
// Created by Marius Meyer on 04.12.19.
//

#include "fft_functionality.hpp"
#include "setup/common_benchmark_io.hpp"

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
                    programSettings->numRepetitions
            });

    //TODO implement actual benchmark execution
    std::complex<HOST_DATA_TYPE>* data;
    posix_memalign(reinterpret_cast<void**>(&data), 64, programSettings->iterations * (1 << LOG_FFT_SIZE) * sizeof(std::complex<HOST_DATA_TYPE>));
    generateInputData(data, programSettings->iterations);

    auto timing =  bm_execution::calculate(config, data, programSettings->iterations, programSettings->inverse);

    auto* verify_data = new std::complex<HOST_DATA_TYPE>[programSettings->iterations * (1 << LOG_FFT_SIZE)];
    generateInputData(verify_data, programSettings->iterations);
    double error = checkFFTResult(verify_data, data, timing->iterations);

    delete [] verify_data;

    printResults(timing);

    return error < 1 ? 0 : 1;
}

