/*
Copyright (c) 2020 Marius Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef SHARED_HPCC_BENCHMAKR_HPP_
#define SHARED_HPCC_BENCHMAKR_HPP_

/* Project's headers */
#include "setup/fpga_setup.hpp"
#include "cxxopts.hpp"
#include "parameters.h"

/* External library headers */
#include "CL/cl.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#define ENTRY_SPACE 15


namespace hpcc_base {

class BaseSettings {

public:

    uint numRepetitions;
    bool useMemoryInterleaving;
    int defaultPlatform;
    int defaultDevice;
    std::string kernelFileName;

    BaseSettings(cxxopts::ParseResult &results) : numRepetitions(results["n"].as<uint>()),
            useMemoryInterleaving(static_cast<bool>(results.count("i"))), 
            defaultPlatform(results["platform"].as<int>()),
            defaultDevice(results["device"].as<int>()),
            kernelFileName(results["f"].as<std::string>()) {}

};

std::ostream& operator<<(std::ostream& os, BaseSettings const& printedBaseSettings){
        return (os    << "Data Type:            " << STR(HOST_DATA_TYPE)
              << std::endl
              << "Repetitions:         " << printedBaseSettings.numRepetitions
              << std::endl
              << "Kernel File:         " << printedBaseSettings.kernelFileName
              << std::endl);
    }

template <class TSettings>
class ExecutionSettings {
public:
    cl::Device device;
    cl::Context context;
    cl::Program program;
    std::shared_ptr<TSettings> programSettings;

    ExecutionSettings(const std::shared_ptr<TSettings> programSettings_, cl::Device device_,cl::Context context_,cl::Program program_): 
                                    programSettings(programSettings_), device(device_), context(context_), program(program_) {}

};

template <class TSettings>
std::ostream& operator<<(std::ostream& os, ExecutionSettings<TSettings> const& printedExecutionSettings){
        return os    << &(printedExecutionSettings.programSettings)
              << "Device:              "
              //<< printedExecutionSettings.device.getInfo<CL_DEVICE_NAME>() 
              << std::endl;
    }

template <class TSettings, class TData, class TOutput>
class HpccFpgaBenchmark {

private:
    bool isSetupExecuted = false;
    ExecutionSettings<TSettings> executionSettings;

protected:

    virtual std::shared_ptr<TData>
    generateInputData(const ExecutionSettings<TSettings> &settings);

    virtual std::shared_ptr<TOutput>
    executeKernel(const ExecutionSettings<TSettings> &settings, TData &data);

    virtual bool
    validateOutputAndPrintError(const ExecutionSettings<TSettings> &settings ,TData &data, const TOutput &output);

    virtual void
    printResults(const ExecutionSettings<TSettings> &settings, const TOutput &output);

    virtual void
    addAdditionalParseOptions(cxxopts::Options &options) {}

    /**
    * Parses and returns program options using the cxxopts library.
    * The parsed parameters are depending on the benchmark that is implementing
    * function.
    * The header file is used to specify a unified interface so it can also be used
    * in the testing binary.
    * cxxopts is used to parse the parameters.
    * @see https://github.com/jarro2783/cxxopts
    * 
    * @param argc Number of input parameters as it is provided by the main function
    * @param argv Strings containing the input parameters as provided by the main function
    * 
    * @return program settings that are created from the given program arguments
    */
    std::shared_ptr<TSettings>
    parseProgramParameters(int argc, char *argv[]) {
        // Defining and parsing program options
        cxxopts::Options options(argv[0], PROGRAM_DESCRIPTION);
        options.add_options()
                ("f,file", "Kernel file name", cxxopts::value<std::string>())
                ("n", "Number of repetitions",
                cxxopts::value<uint>()->default_value(std::to_string(DEFAULT_REPETITIONS)))
                ("i", "Use memory Interleaving")
                ("device", "Index of the device that has to be used. If not given you "\
            "will be asked which device to use if there are multiple devices "\
            "available.", cxxopts::value<int>()->default_value(std::to_string(DEFAULT_DEVICE)))
                ("platform", "Index of the platform that has to be used. If not given "\
            "you will be asked which platform to use if there are multiple "\
            "platforms available.",
                cxxopts::value<int>()->default_value(std::to_string(DEFAULT_PLATFORM)))
                ("h,help", "Print this help");


        addAdditionalParseOptions(&options);
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
        std::shared_ptr<TSettings> sharedSettings(
                new TSettings(result));
        return sharedSettings;
    }

    /**
    * Prints the used configuration to std out before starting the actual benchmark.
    *
    * @param executionSettings The program settings that are parsed from the command line
    *                            using parseProgramParameters and extended by the used OpenCL 
    *                            context, program and device
    */
    void 
    printFinalConfiguration(const ExecutionSettings<TSettings> executionSettings) {
        std::cout << PROGRAM_DESCRIPTION << std::endl;
        std::cout << "Summary:" << std::endl;
        std::cout << executionSettings << std::endl;
    }

public:

    /**
     * @brief Selects and prepares the target device and prints the final configuration
     *          before executing the benchmark
     * 
     * @param argc Number of input parameters as it is provided by the main function
     * @param argv Strings containing the input parameters as provided by the main function
     */
    void 
    setupBenchmark(int argc, char *argv[]) {
        std::shared_ptr<TSettings> programSettings = parseProgramParameters(argc, argv);
        fpga_setup::setupEnvironmentAndClocks();
        cl::Device usedDevice =
            fpga_setup::selectFPGADevice(programSettings->defaultPlatform,
                                            programSettings->defaultDevice);
        cl::Context context = cl::Context(usedDevice);
        cl::Program program = fpga_setup::fpgaSetup(&context, {usedDevice},
                                                &programSettings->kernelFileName);

        executionSettings = ExecutionSettings<TSettings>(programSettings, usedDevice, context, program);

        printFinalConfiguration(executionSettings);
        isSetupExecuted = true;
    }

    bool
    executeBenchmark() {
        if (!isSetupExecuted) {
            std::cerr << "Benchmark execution started without running the benchmark setup!" << std::endl;
            exit(1);
        }
        std::cout << HLINE << "Start benchmark using the given configuration. Generating data..." << std::endl
                << HLINE;
        std::shared_ptr<TData> data = generateInputData(&executionSettings);
        std::cout << HLINE << "Execute benchmar kernel..." << std::endl
                << HLINE;
        std::shared_ptr<TOutput> output =  executeKernel(&executionSettings, &data);

        std::cout << HLINE << "Validate output..." << std::endl
                << HLINE;

        bool validateSuccess = validateOutputAndPrintError(&executionSettings , &data, &output);

        printResults(&executionSettings, &output);

        std::cout << HLINE << "Cleaning up." << std::endl
                << HLINE;

        delete data;
        delete output;

        return validateSuccess;
    }

    HpccFpgaBenchmark(int argc, char *argv[]) {
        setupBenchmark(argc, argv);
    }

};

} // namespace hpcc_base

#endif
