#include "hpcc_settings.hpp"

#ifdef USE_ACCL
#include "setup/fpga_setup_accl.hpp"
#endif
    
    /**
     * @brief Construct a new Base Settings object
     * 
     * @param results The resulting map from parsing the program input parameters
     */
hpcc_base::BaseSettings::BaseSettings(cxxopts::ParseResult &results) : numRepetitions(results["n"].as<uint>()),
#ifdef INTEL_FPGA
            useMemoryInterleaving(static_cast<bool>(results.count("i"))), 
#else
            useMemoryInterleaving(true),
#endif
            skipValidation(static_cast<bool>(results.count("skip-validation"))), 
            defaultPlatform(results["platform"].as<int>()),
            defaultDevice(results["device"].as<int>()),
            kernelFileName(results["f"].as<std::string>()),
#ifdef NUM_REPLICATIONS
            kernelReplications(results.count("r") > 0 ? results["r"].as<uint>() : NUM_REPLICATIONS),
#else
            kernelReplications(results.count("r") > 0 ? results["r"].as<uint>() : 1),
#endif
#ifdef USE_ACCL
            useAcclEmulation(static_cast<bool>(results.count("accl-emulation"))),
            acclProtocol(fpga_setup::acclProtocolStringToEnum(results["accl-protocol"].as<std::string>())),
            acclBufferSize(results["accl-buffer-size"].as<uint>() * 1024),
            acclBufferCount(results["accl-buffer-count"].as<uint>()),
#endif
#ifdef COMMUNICATION_TYPE_SUPPORT_ENABLED
            communicationType(retrieveCommunicationType(results["comm-type"].as<std::string>(), results["f"].as<std::string>())),
#else
            communicationType(retrieveCommunicationType("UNSUPPORTED", results["f"].as<std::string>())),
#endif
            testOnly(static_cast<bool>(results.count("test"))) {}

/**
 * @brief Get a map of the settings. This map will be used to print the final configuration.
 *          Derived classes should override it to add additional configuration options
 * 
 * @return std::map<std::string,std::string> 
 */
std::map<std::string,std::string> 
hpcc_base::BaseSettings::getSettingsMap() {
    int mpi_size = 0;
#ifdef _USE_MPI_
     MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    std::string str_mpi_ranks = "None";
    if (mpi_size > 0) {
        str_mpi_ranks = std::to_string(mpi_size);
    }
    return {{"Repetitions", std::to_string(numRepetitions)}, {"Kernel Replications", std::to_string(kernelReplications)}, 
            {"Kernel File", kernelFileName}, {"MPI Ranks", str_mpi_ranks}, {"Test Mode", testOnly ? "Yes" : "No"},
            {"Communication Type", commToString(communicationType)}};
}