#include "transpose_benchmark.hpp"

using namespace transpose;

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Setup benchmark
#ifdef USE_OCL_HOST
    TransposeBenchmark<cl::Device, cl::Context, cl::Program> bm(argc, argv);
#else
    TransposeBenchmark<xrt::device, bool, xrt::uuid> bm(argc, argv);
#endif
    bool success = bm.executeBenchmark();
    if (success) {
        return 0;
    }
    else {
        return 1;
    }
}

