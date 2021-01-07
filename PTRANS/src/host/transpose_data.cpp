
#include "transpose_data.hpp"

transpose::TransposeProgramSettings::TransposeProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * results["b"].as<uint>()),
    blockSize(results["b"].as<uint>()), dataHandlerIdentifier(results["handler"].as<std::string>()),
    distributeBuffers(results["distribute-buffers"].count() > 0) {

}

std::map<std::string, std::string>
transpose::TransposeProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        map["Dist. Buffers"] = distributeBuffers ? "Yes" : "No";
        map["Data Handler"] = dataHandlerIdentifier;
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
#else
        posix_memalign(reinterpret_cast<void **>(&A), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        posix_memalign(reinterpret_cast<void **>(&B), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        posix_memalign(reinterpret_cast<void **>(&result), 64,
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
#else
        free(A);
        free(B);
        free(result);
#endif
    }
}
