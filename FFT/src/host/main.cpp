//
// Created by Marius Meyer on 04.12.19.
//

#include "fft_benchmark.hpp"

using namespace fft;

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Setup benchmark
    FFTBenchmark bm(argc, argv);
    bool success = bm.executeBenchmark();
    if (success) {
        return 0;
    }
    else {
        return 1;
    }
}
