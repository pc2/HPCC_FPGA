#include "hpcc_settings.hpp"

#ifdef USE_ACCL
#include "setup/fpga_setup_accl.hpp"
#endif
#ifdef USE_XRT
#include "setup/fpga_setup_udp.hpp"
#endif

/**
 * @brief Construct a new Base Settings object
 *
 * @param results The resulting map from parsing the program input parameters
 */
hpcc_base::BaseSettings::BaseSettings(cxxopts::ParseResult &results)
    : numRepetitions(results["n"].as<uint>()),
#ifdef INTEL_FPGA
      useMemoryInterleaving(static_cast<bool>(results.count("i"))),
#else
      useMemoryInterleaving(true),
#endif
      skipValidation(static_cast<bool>(results.count("skip-validation"))),
      defaultPlatform(results["platform"].as<int>()), defaultDevice(results["device"].as<int>()),
      kernelFileName(results["f"].as<std::string>()), dumpfilePath(results["dump-json"].as<std::string>()),
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
      acclRecvBufferMemBanks(results["accl-recv-banks"].as<std::vector<int>>()),
      acclDefaultBank(results["accl-default-bank"].as<int>()),
#endif
#ifdef COMMUNICATION_TYPE_SUPPORT_ENABLED
      communicationType(
          retrieveCommunicationType(results["comm-type"].as<std::string>(), results["f"].as<std::string>())),
#else
      communicationType(retrieveCommunicationType("UNSUPPORTED", results["f"].as<std::string>())),
#endif
      testOnly(static_cast<bool>(results.count("test")))
{
}

/**
 * @brief Get a map of the settings. This map will be used to print the final configuration.
 *          Derived classes should override it to add additional configuration options
 *
 * @return std::map<std::string,std::string>
 */
std::map<std::string, std::string> hpcc_base::BaseSettings::getSettingsMap()
{
    int mpi_size = 0;
#ifdef _USE_MPI_
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    std::string str_mpi_ranks = "None";
    if (mpi_size > 0) {
        str_mpi_ranks = std::to_string(mpi_size);
    }
#ifdef USE_ACCL
    std::stringstream accl_recv_banks;
    for (auto &b : acclRecvBufferMemBanks) {
        accl_recv_banks << b << ",";
    }
#endif
    return {{"Repetitions", std::to_string(numRepetitions)},
            {"Kernel Replications", std::to_string(kernelReplications)},
            {"Kernel File", kernelFileName},
            {"MPI Ranks", str_mpi_ranks},
            {"Test Mode", testOnly ? "Yes" : "No"},
            {"Communication Type", commToString(communicationType)}
#ifdef USE_ACCL
            ,
            {"ACCL Protocol", fpga_setup::acclEnumToProtocolString(acclProtocol)},
            {"ACCL Recv. Banks", accl_recv_banks.str()},
            {"ACCL Default Bank", std::to_string(acclDefaultBank)},
            {"ACCL Buffer Size", std::to_string(acclBufferSize) + "KB"},
            {"ACCL Buffer Count", std::to_string(acclBufferCount)},
            {"ACCL Emulation", useAcclEmulation ? "Yes" : "No"}
#endif
    };
}
