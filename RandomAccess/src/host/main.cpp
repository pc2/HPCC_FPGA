#include "random_access_benchmark.hpp"

using namespace random_access;

/**
The program entry point
*/
int
main(int argc, char *argv[]) {
    // Setup benchmark
    RandomAccessBenchmark bm(argc, argv);
    bool success = bm.executeBenchmark();
    if (success) {
        return 0;
    }
    else {
        return 1;
    }
}

