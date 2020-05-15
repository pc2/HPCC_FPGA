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
#ifndef SHARED_HPCC_BENCHMARK_HPP_
#define SHARED_HPCC_BENCHMARK_HPP_

/* Project's headers */
#include "setup/fpga_setup.hpp"
#include "cxxopts.hpp"
#include "parameters.h"

/* External library headers */
#include "CL/cl.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#define ENTRY_SPACE 15

/**
 * @brief Contains all classes and functions that are used as basis
 *          for all benchmarks.
 * 
 */
namespace hpcc_base {

/**
 * @brief This class should be derived and extended for every benchmark.
 *          It is a pure data object containing the benchmark settings that are
 *          used to execute the benchmark kernel.
 * 
 *  DERIVED CLASSES NEED TO IMPLEMENT A operator<< METHOD! 
 *       
 */
class BaseSettings {

public:

    /**
     * @brief Number of times the kernel execution will be repeated
     * 
     */
    uint numRepetitions;

    /**
     * @brief Boolean showing if memory interleaving is used that is 
     *          triggered from the host side (Intel specific)
     * 
     */
    bool useMemoryInterleaving;

    /**
     * @brief The default platform that should be used for execution. 
     *          A number representing the index in the list of available platforms
     * 
     */
    int defaultPlatform;

    /**
     * @brief The default device that should be used for execution. 
     *          A number representing the index in the list of available devices
     * 
     */
    int defaultDevice;

    /**
     * @brief Path to the kernel file that is used for execution
     * 
     */
    std::string kernelFileName;

    /**
     * @brief Construct a new Base Settings object
     * 
     * @param results The resulting map from parsing the program input parameters
     */
    BaseSettings(cxxopts::ParseResult &results) : numRepetitions(results["n"].as<uint>()),
            useMemoryInterleaving(static_cast<bool>(results.count("i"))), 
            defaultPlatform(results["platform"].as<int>()),
            defaultDevice(results["device"].as<int>()),
            kernelFileName(results["f"].as<std::string>()) {}

    virtual std::map<std::string,std::string> getSettingsMap() {
        return {{"Repetitions", std::to_string(numRepetitions)}, {"Kernel File", kernelFileName}};
    }

};


/**
 * @brief Settings class that is containing the program settings together with
 *          additional information about the OpenCL runtime
 * 
 * @tparam TSettings The program settings class that should be used (Must derive from BaseSettings)
 */
template <class TSettings>
class ExecutionSettings {
public:

    /**
     * @brief The OpenCL device that should be used for execution
     * 
     */
    cl::Device device;

    /**
     * @brief The OpenCL context that should be used for execution
     * 
     */
    cl::Context context;

    /**
     * @brief The OpenCL program that contains the benchmark kernel
     * 
     */
    cl::Program program;

    /**
     * @brief Pointer to the additional program settings
     * 
     */
    std::shared_ptr<TSettings> programSettings;

    /**
     * @brief Construct a new Execution Settings object
     * 
     * @param programSettings_ Pointer to an existing program settings object that is derived from BaseSettings
     * @param device_ Used OpenCL device
     * @param context_ Used OpenCL context
     * @param program_ Used OpenCL program
     */
    ExecutionSettings(const std::shared_ptr<TSettings> programSettings_, cl::Device device_,cl::Context context_,cl::Program program_): 
                                    programSettings(programSettings_), device(device_), context(context_), program(program_) {}
    
    /**
     * @brief Construct a new Execution Settings object from an Execution Settings object
     * 
     * @param s The object to copy
     */
    ExecutionSettings(ExecutionSettings<TSettings> *s) : ExecutionSettings(s->programSettings, s->device, s->context, s->program) {}

};

/**
 * @brief Base benchmark class. Every benchmark should be derived from this class and implement its abstract methods.
 * 
 * @tparam TSettings Class used to represent the program settings of the benchmark
 * @tparam TData Class used to represent the benchmark input and output data
 * @tparam TOutput Class representing the measurements like timings etc
 */
template <class TSettings, class TData, class TOutput>
class HpccFpgaBenchmark {

private:
    /**
     * @brief Is set by setupBenchmark() to make sure the benchmark is not 
     *          executed before the setup is run
     * 
     */
    bool isSetupExecuted = false;

    /**
     * @brief The used execution settings that will be generated by setupBenchmark()
     * 
     */
    std::shared_ptr<ExecutionSettings<TSettings>> executionSettings;

protected:

    /**
     * @brief Add additional options to the program parameter parser
     * 
     * @param options The options object that will be used to parse the input parameters
     */
    virtual void
    addAdditionalParseOptions(cxxopts::Options &options) {}

public:

    /**
     * @brief Allocate and initiate the input data for the kernel
     * 
     * @param settings The used execution settings
     * @return std::shared_ptr<TData> A data class containing the initialized data
     */
    virtual std::shared_ptr<TData>
    generateInputData(const ExecutionSettings<TSettings> &settings) = 0;

    /**
     * @brief Execute the benchmark kernel and measure performance
     * 
     * @param settings The used execution settings
     * @param data The initialized data for the kernel. It will be replaced by the kernel output for validation
     * @return std::shared_ptr<TOutput> A data class containing the measurement results of the execution
     */
    virtual std::shared_ptr<TOutput>
    executeKernel(const ExecutionSettings<TSettings> &settings, TData &data) = 0;

    /**
     * @brief Validate the output of the execution
     * 
     * @param settings The used execution settings
     * @param data The output data after kernel execution
     * @param output The measurement data of the kernel execution
     * @return true If the validation is a success.
     * @return false If the validation failed
     */
    virtual bool
    validateOutputAndPrintError(const ExecutionSettings<TSettings> &settings ,TData &data, const TOutput &output) = 0;

    /**
     * @brief Prints the measurement results of the benchmark to std::cout
     * 
     * @param settings The used execution settings
     * @param output  The measurement data of the kernel execution
     */
    virtual void
    printResults(const ExecutionSettings<TSettings> &settings, const TOutput &output) = 0;

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


        addAdditionalParseOptions(options);
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
    printFinalConfiguration(ExecutionSettings<TSettings> const& executionSettings) {
        std::cout << PROGRAM_DESCRIPTION << std::endl;
        std::cout << "Summary:" << std::endl;
        std::cout << executionSettings << std::endl;
    }

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
        cl::Device usedDevice =
            cl::Device(fpga_setup::selectFPGADevice(programSettings->defaultPlatform,
                                            programSettings->defaultDevice));
        cl::Context context = cl::Context(usedDevice);
        cl::Program program = fpga_setup::fpgaSetup(&context, {usedDevice},
                                                &programSettings->kernelFileName);

        executionSettings = std::make_shared<ExecutionSettings<TSettings>>(new ExecutionSettings<TSettings>(programSettings, usedDevice, context, program));

        printFinalConfiguration(*executionSettings);
        isSetupExecuted = true;
    }

    /**
     * @brief Execute the benchmark. This includes the initialization of the 
     *          input data, exectuon of the kernel, validation and printing the result
     * 
     * @return true If the validation is a success
     * @return false If the validation fails
     */
    bool
    executeBenchmark() {
        if (!isSetupExecuted) {
            std::cerr << "Benchmark execution started without running the benchmark setup!" << std::endl;
            exit(1);
        }
        std::cout << HLINE << "Start benchmark using the given configuration. Generating data..." << std::endl
                << HLINE;
        std::shared_ptr<TData> data = generateInputData(*executionSettings);
        std::cout << HLINE << "Execute benchmark kernel..." << std::endl
                << HLINE;
        std::shared_ptr<TOutput> output =  executeKernel(*executionSettings, *data);

        std::cout << HLINE << "Validate output..." << std::endl
                << HLINE;

        bool validateSuccess = validateOutputAndPrintError(*executionSettings , *data, *output);

        printResults(*executionSettings, *output);

        return validateSuccess;
    }

    /**
     * @brief Get the Execution Settings object for further modifications (Mainly for testing purposes)
     * 
     * @return ExecutionSettings& The execution settings object
     */
    ExecutionSettings<TSettings>& getExecutionSettings() {
        return *executionSettings;
    }

    HpccFpgaBenchmark() {
        fpga_setup::setupEnvironmentAndClocks();
    }

};

/**
 * @brief Prints the execution settings to an output stream
 * 
 * @tparam TSettings The program settings class used to create the execution settings
 * @param os The output stream
 * @param printedExecutionSettings The execution settings that have to be printed to the stream
 * @return std::ostream& The output stream after the execution settings are piped in
 */
template <class TSettings>
std::ostream& operator<<(std::ostream& os, ExecutionSettings<TSettings> const& printedExecutionSettings){
        std::string device_name;
        printedExecutionSettings.device.getInfo(CL_DEVICE_NAME, &device_name);
        for (auto k : printedExecutionSettings.programSettings->getSettingsMap()) {
            os   << std::setw(2 * ENTRY_SPACE) << std::left<< k.first << k.second << std::endl;
        }
        os  << std::setw(2 * ENTRY_SPACE) << std::left << "Device"  << device_name << std::endl;
        return os;
}

} // namespace hpcc_base

#endif
