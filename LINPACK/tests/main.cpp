/*
Copyright (c) 2020 Marius Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/* Project's headers */
#include "linpack_benchmark.hpp"

#include "gtest/gtest.h"
#include "CL/cl.hpp"

#ifdef _USE_MPI_
#include "mpi.h"

class MPIEnvironment : public ::testing::Environment {
public:
    MPIEnvironment(int* argc, char** argv[]) {
        MPI_Init(argc, argv);
    }

    ~MPIEnvironment() override {
        MPI_Finalize();
    }
};
#endif

using namespace linpack;

std::unique_ptr<LinpackBenchmark> bm;

/**
The program entry point for the unit tests
*/
int
main(int argc, char *argv[]) {

    std::cout << "THIS BINARY EXECUTES UNIT TESTS FOR THE FOLLOWING BENCHMARK:" << std::endl << std::endl;

    ::testing::InitGoogleTest(&argc, argv);

    bm = std::unique_ptr<LinpackBenchmark>(new LinpackBenchmark(argc, argv));

#ifdef _USE_MPI_
    ::testing::Environment* const mpi_env =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment(&argc, &argv));
#endif

    bool result = RUN_ALL_TESTS();

    bm = nullptr;

    return result;

}

