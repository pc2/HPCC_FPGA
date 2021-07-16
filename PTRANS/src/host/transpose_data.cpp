
#include "transpose_data.hpp"
#include "data_handlers/data_handler_types.h"
#include "fpga_execution/communication_types.h"

transpose::TransposeProgramSettings::TransposeProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * results["b"].as<uint>()),
    blockSize(results["b"].as<uint>()), dataHandlerIdentifier(transpose::data_handler::stringToHandler(results["handler"].as<std::string>())),
    distributeBuffers(results["distribute-buffers"].count() > 0), communicationType(transpose::fpga_execution::stringToComm(results["connectivity"].as<std::string>())) {

        // auto detect data communication type if required
        if (communicationType == transpose::fpga_execution::CommunicationType::automatic) {
            if (kernelFileName.find("_" + transpose::fpga_execution::commToString(transpose::fpga_execution::CommunicationType::pcie_mpi)) != kernelFileName.npos) {
                communicationType = transpose::fpga_execution::CommunicationType::pcie_mpi;
            }
            else if (kernelFileName.find("_" + transpose::fpga_execution::commToString(transpose::fpga_execution::CommunicationType::intel_external_channels)) != kernelFileName.npos) {
                communicationType = transpose::fpga_execution::CommunicationType::intel_external_channels;
            }
            if (communicationType == transpose::fpga_execution::CommunicationType::automatic) {
                throw std::runtime_error("Communication could not be detected from kernel file name!");
            }
        }

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
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        map["Dist. Buffers"] = distributeBuffers ? "Yes" : "No";
        map["Data Handler"] = transpose::data_handler::handlerToString(dataHandlerIdentifier);
        map["Communication Type"] = transpose::fpga_execution::commToString(communicationType);
        return map;
}

transpose::TransposeData::TransposeData(cl::Context context, uint block_size, uint y_size) : context(context), 
                                                                                numBlocks(y_size), blockSize(block_size) {
    if (numBlocks * blockSize > 0) {
#ifdef USE_SVM
        A = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 1024));
        B = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 1024));
        result = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 1024));
        exchange = reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 1024));
#else
        posix_memalign(reinterpret_cast<void **>(&A), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        posix_memalign(reinterpret_cast<void **>(&B), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        posix_memalign(reinterpret_cast<void **>(&result), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        posix_memalign(reinterpret_cast<void **>(&exchange), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
#endif
    }
}

transpose::TransposeData::~TransposeData() {
    if (numBlocks * blockSize > 0) {
#ifdef USE_SVM
        clSVMFree(context(), reinterpret_cast<void*>(A));});
        clSVMFree(context(), reinterpret_cast<void*>(B));});
        clSVMFree(context(), reinterpret_cast<void*>(result));});
        clSVMFree(context(), reinterpret_cast<void*>(exchange));});
#else
        free(A);
        free(B);
        free(result);
        free(exchange);
#endif
    }
}
