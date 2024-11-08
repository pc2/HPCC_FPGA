#include <cmath>

#include "transpose_data.hpp"
#include "data_handlers/data_handler_types.h"
#include "communication_types.hpp"

transpose::TransposeProgramSettings::TransposeProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * results["b"].as<uint>()),
    blockSize(results["b"].as<uint>()), dataHandlerIdentifier(transpose::data_handler::stringToHandler(results["handler"].as<std::string>())),
    distributeBuffers(results["distribute-buffers"].count() > 0), p(results["p"].as<uint>()), copyA(results["copy-a"].count() > 0),
    useAcclStreams(results["accl-stream"].count() > 0) {

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
        int mpi_comm_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
        // calculate the row and column of the MPI rank in the torus 
        if (mpi_comm_size % p != 0) {
            throw std::runtime_error("MPI Comm size not dividable by P=" + std::to_string(p) + "!");
        } 
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        map["Dist. Buffers"] = distributeBuffers ? "Yes" : "No";
        map["Data Handler"] = transpose::data_handler::handlerToString(dataHandlerIdentifier);
        map["FPGA Torus"] = "P=" + std::to_string(p) + " ,Q=" + std::to_string(mpi_comm_size / p);
        return map;
}

