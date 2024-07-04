//
// Created by Marius Meyer on 04.12.19.
//

#include "linpack_benchmark.hpp"

using namespace linpack;

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Setup benchmark
#ifdef USE_OCL_HOST
    LinpackBenchmark<cl::Device, cl::Context, cl::Program> bm(argc, argv);
#endif
#ifdef USE_XRT_HOST
#ifndef USE_ACCL
    LinpackBenchmark<xrt::device, bool, xrt::uuid> bm(argc, argv);
#else
    LinpackBenchmark<xrt::device, fpga_setup::ACCLContext, xrt::uuid> bm(argc, argv);
#endif
#endif
    bool success = bm.executeBenchmark();
    if (success) {
        return 0;
    }
    else {
        return 1;
    }
}

