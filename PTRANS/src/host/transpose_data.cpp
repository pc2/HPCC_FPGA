#include <cmath>

#include "transpose_data.hpp"
#include "data_handlers/data_handler_types.h"
#include "communication_types.hpp"

transpose::TransposeProgramSettings::TransposeProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * results["b"].as<uint>()),
    blockSize(results["b"].as<uint>()), dataHandlerIdentifier(transpose::data_handler::stringToHandler(results["handler"].as<std::string>())),
    distributeBuffers(results["distribute-buffers"].count() > 0), p(results["p"].as<uint>()) {

        // auto detect data distribution type if required
        if (dataHandlerIdentifier == transpose::data_handler::DataHandlerType::automatic) {
            if (kernelFileName.find("_"+ transpose::data_handler::handlerToString(transpose::data_handler::DataHandlerType::diagonal) +"_") != kernelFileName.npos) {
                dataHandlerIdentifier = transpose::data_handler::DataHandlerType::diagonal;
            }
            else if (kernelFileName.find("_"+ transpose::data_handler::handlerToString(transpose::data_handler::DataHandlerType::pq) + "_") != kernelFileName.npos) {
                dataHandlerIdentifier = transpose::data_handler::DataHandlerType::pq;
            }
            if (dataHandlerIdentifier == transpose::data_handler::DataHandlerType::automatic) {
                throw std::runtime_error("Required data distribution could not be detected from kernel file name!");
            }
        }
}

std::map<std::string, std::string>
transpose::TransposeProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        int mpi_size;
#ifdef _USE_MPI_
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        map["Dist. Buffers"] = distributeBuffers ? "Yes" : "No";
        map["Data Handler"] = transpose::data_handler::handlerToString(dataHandlerIdentifier);
        return map;
}

