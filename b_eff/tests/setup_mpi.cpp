#include "gtest/gtest.h"
#include "mpi.h"

class MPIEnvironment : public ::testing::Environment {
public:
    ~MPIEnvironment() override {}

    // Override this to define how to set up the environment.
    void SetUp() override {
        MPI_Init(NULL, NULL);
    }

    // Override this to define how to tear down the environment.
    void TearDown() override {
        MPI_Finalize();
    }
};

::testing::Environment* const mpi_env =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment);