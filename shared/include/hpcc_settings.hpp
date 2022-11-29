#ifndef HPCC_BASE_SETTINGS_H_
#define HPCC_BASE_SETTINGS_H_

#include "cxxopts.hpp"
#include "parameters.h"
#include "communication_types.hpp"

#ifdef _USE_MPI_
#include "mpi.h"
#endif

#ifdef USE_ACCL
#include "accl.hpp"
#endif

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
     * @brief Boolean showing if the output data of the benchmark kernel
     *          should be validated or not
     * 
     */
    bool skipValidation;

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
     * @brief Number of times the kernel is replicated
     * 
     */
    uint kernelReplications;

    /**
     * @brief Only test the given configuration. Do not execute the benchmarks
     * 
     */
    bool testOnly;

    /**
     * @brief Type of inter-FPGA communication used
     * 
     */
    CommunicationType communicationType;

#ifdef USE_ACCL
    /**
     * @brief Use ACCL emulation constructor instead of hardware execution
     */
    bool useAcclEmulation;

    /**
     * @brief Used ACCL network stack
     * 
     */
    ACCL::networkProtocol acclProtocol;

    /**
     * @brief Size of the ACCL buffers in bytes
     * 
     */
    uint acclBufferSize;

    /**
     * @brief Number of ACCL buffers to use
     * 
     */
    uint acclBufferCount;
#endif

    /**
     * @brief Construct a new Base Settings object
     * 
     * @param results The resulting map from parsing the program input parameters
     */
    BaseSettings(cxxopts::ParseResult &results);

    /**
     * @brief Get a map of the settings. This map will be used to print the final configuration.
     *          Derived classes should override it to add additional configuration options
     * 
     * @return std::map<std::string,std::string> 
     */
    virtual std::map<std::string,std::string> getSettingsMap();

};

/**
 * @brief Settings class that is containing the program settings together with
 *          additional information about the OpenCL runtime
 * 
 * @tparam TSettings The program settings class that should be used (Must derive from BaseSettings)
 */
template <class TSettings, class TDevice, class TContext, class TProgram, typename =
typename std::enable_if<std::is_base_of<hpcc_base::BaseSettings, TSettings>::value>::type>
class ExecutionSettings {
public:

    /**
     * @brief Pointer to the additional program settings
     * 
     */
    std::unique_ptr<TSettings> programSettings;

    /**
     * @brief The OpenCL device that should be used for execution
     * 
     */
    std::unique_ptr<TDevice> device;

    /**
     * @brief The OpenCL context that should be used for execution
     * 
     */
    std::unique_ptr<TContext> context;

    /**
     * @brief The OpenCL program that contains the benchmark kernel
     * 
     */
    std::unique_ptr<TProgram> program;

    /**
     * @brief Construct a new Execution Settings object
     * 
     * @param programSettings_ Pointer to an existing program settings object that is derived from BaseSettings
     * @param device_ Used OpenCL device
     * @param context_ Used OpenCL context
     * @param program_ Used OpenCL program
     */
    ExecutionSettings(std::unique_ptr<TSettings> programSettings_, std::unique_ptr<TDevice> device_, 
                        std::unique_ptr<TContext> context_, std::unique_ptr<TProgram> program_
                        
                        ): 
                                    programSettings(std::move(programSettings_)), device(std::move(device_)), 
                                    context(std::move(context_)), program(std::move(program_))                                    
                                             {}

    /**
     * @brief Destroy the Execution Settings object. Used to specify the order the contained objects are destroyed 
     *         to prevent segmentation faults during exit.
     * 
     */
    ~ExecutionSettings() {
        program = nullptr;
        context = nullptr;
        device = nullptr;
        programSettings = nullptr;
    }

};

}

#endif