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

#include "gtest/gtest.h"
#include "CL/cl.hpp"

#ifdef _USE_MPI_
#include "mpi.h"

class MPIEnvironment : public ::testing::Environment {
public:
    MPIEnvironment(int* argc, char** argv[]) {
        int isMPIInitialized;
        MPI_Initialized(&isMPIInitialized);
        if (!isMPIInitialized) {
            MPI_Init(argc, argv);
        }
    }

    ~MPIEnvironment() override {
        int isMPIFinalized;
        MPI_Finalized(&isMPIFinalized);
        if (!isMPIFinalized) {
            MPI_Finalize();
        }
    }
};
#endif

extern void use_hpcc_base_lib();

int global_argc;
char** global_argv;

/**
The program entry point for the unit tests
*/
int
main(int argc, char *argv[]) {

    std::cout << "THIS BINARY EXECUTES UNIT TESTS FOR THE FOLLOWING BENCHMARK:" << std::endl << std::endl;

    ::testing::InitGoogleTest(&argc, argv);

#ifdef _USE_MPI_
    ::testing::Environment* const mpi_env =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment(&argc, &argv));
#endif

    global_argc = argc;
    global_argv = argv;

    use_hpcc_base_lib();

    bool result = RUN_ALL_TESTS();

    return result;

}
