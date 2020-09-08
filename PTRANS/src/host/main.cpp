#include "transpose_benchmark.hpp"

using namespace transpose;

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Setup benchmark
    TransposeBenchmark bm(argc, argv);
    bool success = bm.executeBenchmark();
    if (success) {
        return 0;
    }
    else {
        return 1;
    }
}

