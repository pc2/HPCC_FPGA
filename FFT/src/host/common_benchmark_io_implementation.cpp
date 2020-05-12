
#include "cxxopts.hpp"
#include "parameters.h"
#include "setup/common_benchmark_io.hpp"

/**
Parses and returns program options using the cxxopts library.
Supports the following parameters:
    - file name of the FPGA kernel file (-f,--file)
    - number of repetitions (-n)
    - number of kernel replications (-r)
    - data size (-d)
    - use memory interleaving
@see https://github.com/jarro2783/cxxopts

@return program settings that are created from the given program arguments
*/
std::shared_ptr<ProgramSettings>
parseProgramParameters(int argc, char *argv[]) {
    // Defining and parsing program options
    cxxopts::Options options(argv[0], PROGRAM_DESCRIPTION);
    options.add_options()
            ("f,file", "Kernel file name", cxxopts::value<std::string>())
            ("n", "Number of repetitions",
             cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_REPETITIONS)))
            ("i", "Multiplier for the used data size that will be i * FFT_SIZE",
             cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_ITERATIONS)))
            ("inverse", "If set, the inverse FFT is calculated instead")
            ("device", "Index of the device that has to be used. If not given you "\
        "will be asked which device to use if there are multiple devices "\
        "available.", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_DEVICE)))
            ("platform", "Index of the platform that has to be used. If not given "\
        "you will be asked which platform to use if there are multiple "\
        "platforms available.",
             cxxopts::value<int>()->default_value(std::to_string(DEFAULT_PLATFORM)))
            ("h,help", "Print this help");
    cxxopts::ParseResult result = options.parse(argc, argv);
    
    if (result.count("h")) {
        // Just print help when argument is given
        std::cout << options.help() << std::endl;
        exit(0);
    }
    // Check parsed options and handle special cases
    if (result.count("f") <= 0) {
        // Path to the kernel file is mandatory - exit if not given!
        std::cerr << "Kernel file must be given! Aborting" << std::endl;
        std::cout << options.help() << std::endl;
        exit(1);
    }

    // Create program settings from program arguments
    std::shared_ptr<ProgramSettings> sharedSettings(
            new ProgramSettings{result["n"].as<uint>(),
                                result["i"].as<uint>(),
                                static_cast<bool>(result.count("inverse")),
                                result["platform"].as<int>(),
                                result["device"].as<int>(),
                                result["f"].as<std::string>()});
    return sharedSettings;
}

/**
 * Prints the used configuration to std out before starting the actual benchmark.
 *
 * @param programSettings The program settings retrieved from the command line
 * @param device The device used for execution
 */
void printFinalConfiguration(const std::shared_ptr<ProgramSettings> &programSettings,
                             const cl::Device &device) {// Give setup summary
    std::cout << PROGRAM_DESCRIPTION << std::endl << HLINE;
    std::cout << "Summary:" << std::endl
              << "FFT Size:            " << (1 << LOG_FFT_SIZE)
              << std::endl
              << "Data Size:           " << programSettings->iterations << " * FFT Size * sizeof("
              << STR(HOST_DATA_TYPE)
              << ") = " << static_cast<double>((1 << LOG_FFT_SIZE) * programSettings->iterations * sizeof(HOST_DATA_TYPE)) << " Byte"
              << std::endl
              << "Repetitions:         " << programSettings->numRepetitions
              << std::endl
              << "Kernel file:         " << programSettings->kernelFileName
              << std::endl;
    std::cout << "Device:              "
              << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << HLINE
              << "Start benchmark using the given configuration." << std::endl
              << HLINE;
}
