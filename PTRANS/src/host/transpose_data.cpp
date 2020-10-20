
#include "transpose_data.hpp"

transpose::TransposeProgramSettings::TransposeProgramSettings(cxxopts::ParseResult &results) : hpcc_base::BaseSettings(results),
    matrixSize(results["m"].as<uint>() * results["b"].as<uint>()),
    blockSize(results["b"].as<uint>()), dataHandlerIdentifier(results["handler"].as<std::string>()) {

}

std::map<std::string, std::string>
transpose::TransposeProgramSettings::getSettingsMap() {
        auto map = hpcc_base::BaseSettings::getSettingsMap();
        map["Matrix Size"] = std::to_string(matrixSize);
        map["Block Size"] = std::to_string(blockSize);
        map["Data Handler"] = dataHandlerIdentifier;
        return map;
}

transpose::TransposeData::TransposeData(cl::Context context, uint block_size, uint y_size, uint numReplications) : context(context) {
    numBlocks = y_size;
    for (int r = 0; r < numReplications; r++) {
#ifdef USE_SVM
        A.push_back(reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 1024)));
        B.push_back(reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 1024)));
        result.push_back(reinterpret_cast<HOST_DATA_TYPE*>(
                            clSVMAlloc(context(), 0 ,
                            block_size * block_size * y_size * sizeof(HOST_DATA_TYPE), 1024)));
#else
        HOST_DATA_TYPE *tmpA, *tmpB, *tmpResult;
        posix_memalign(reinterpret_cast<void **>(&tmpA), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        posix_memalign(reinterpret_cast<void **>(&tmpB), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        posix_memalign(reinterpret_cast<void **>(&tmpResult), 64,
                    sizeof(HOST_DATA_TYPE) * block_size * block_size * y_size);
        A.push_back(tmpA);
        B.push_back(tmpB);
        result.push_back(tmpResult);
#endif
    }
}

transpose::TransposeData::~TransposeData() {
#ifdef USE_SVM
    std::for_each(A.begin(), A.end(),[](HOST_DATA_TYPE* a){ clSVMFree(context(), reinterpret_cast<void*>(a));});
    std::for_each(B.begin(), B.end(),[](HOST_DATA_TYPE* b){ clSVMFree(context(), reinterpret_cast<void*>(b));});
    std::for_each(result.begin(), result.end(),[](HOST_DATA_TYPE* r){ clSVMFree(context(), reinterpret_cast<void*>(r));});
#else
    std::for_each(A.begin(), A.end(),[](HOST_DATA_TYPE* a){ free(a);});
    std::for_each(B.begin(), B.end(),[](HOST_DATA_TYPE* b){ free(b);});
    std::for_each(result.begin(), result.end(),[](HOST_DATA_TYPE* r){ free(r);});
#endif
}
