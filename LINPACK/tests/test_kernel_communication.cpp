#include "gtest/gtest.h"
#include "parameters.h"
#include "test_program_settings.h"
#include "linpack_benchmark.hpp"


#define BLOCK_SIZE (1 << LOCAL_MEM_BLOCK_LOG)
#define CHUNK (1 << REGISTER_BLOCK_LOG)


class LinpackKernelCommunicationTest : public testing::Test {

public:
    std::unique_ptr<linpack::LinpackBenchmark> bm;
    std::unique_ptr<linpack::LinpackData> data;
    const unsigned numberOfChannels = 4;
    const std::string channelOutName = "kernel_output_ch";
    const std::string channelInName = "kernel_input_ch";

    void SetUp() override {
        bm = std::unique_ptr<linpack::LinpackBenchmark>(new linpack::LinpackBenchmark(global_argc, global_argv));
        bm->getExecutionSettings().programSettings->isDiagonallyDominant = true;
        bm->getExecutionSettings().programSettings->matrixSize = BLOCK_SIZE;
        data = bm->generateInputData();
        setupExternalChannelFiles();
    }

    /**
     * @brief Setup the external channels files for the execution of a benchmark kernel
     * 
     */
    void
    setupExternalChannelFiles() {
        for (int i=0; i < numberOfChannels; i++) {
            std::string fname = channelOutName + std::to_string(i);
            std::remove(fname.c_str());
            std::ofstream fs;
            fs.open(fname, std::ofstream::out | std::ofstream::trunc);
            fs.close();
            std::remove((channelInName + std::to_string(i%2 ? i-1 : i+1)).c_str());
            symlink(fname.c_str(), (channelInName + std::to_string(i%2 ? i-1 : i+1)).c_str());
        }
    }

    /**
     * @brief Get the Data sent over an external channel.
     * 
     * @param channel_id Id of the external channel. It is assumed to be conntected in the order 0->Top,1->Right,2->Bottom,3->Left.
     *                  So channel 0 will be connected to the channel 2 of the FPGA above the current FPGA in the 2D Torus. 
     * @return std::vector<HOST_DATA_TYPE> The data that is contained in the output file of the channel
     */
    std::vector<HOST_DATA_TYPE>
    getDataFromExternalChannel(uint channel_id) {
        std::string fname = channelOutName + std::to_string(channel_id);
        std::ifstream fs;
        fs.open(fname, std::ifstream::binary | std::ifstream::in);
        HOST_DATA_TYPE value;
        std::vector<HOST_DATA_TYPE> values;
        while (fs.read(reinterpret_cast<char*>(&value), sizeof(HOST_DATA_TYPE))) {
            values.push_back(value);
        }
        return values;
    }

};


class LinpackKernelCommunicationTestLU : public LinpackKernelCommunicationTest {

    void SetUp() override {
        LinpackKernelCommunicationTest::SetUp();
        executeKernel();
    }

    void executeKernel() {
        int err;
        cl::CommandQueue compute_queue(*bm->getExecutionSettings().context, *bm->getExecutionSettings().device, 0, &err);
        cl::Buffer buffer(*(bm->getExecutionSettings().context), CL_MEM_READ_WRITE,
                                            sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize);
        cl::Kernel kernel(*bm->getExecutionSettings().program, "lu", &err);

        err = kernel.setArg(0, buffer);

        compute_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
        compute_queue.enqueueTask(kernel);
        compute_queue.finish();
        compute_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(HOST_DATA_TYPE)*bm->getExecutionSettings().programSettings->matrixSize*bm->getExecutionSettings().programSettings->matrixSize, data->A);
    }
};


TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalResultisCorrect) {
    linpack::gesl_ref_nopvt(data->A, data->b, bm->getExecutionSettings().programSettings->matrixSize,bm->getExecutionSettings().programSettings->matrixSize);
    EXPECT_TRUE(bm->validateOutputAndPrintError(*data));

}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToRightCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(1);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_left.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToLeftCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(3);

    EXPECT_EQ(data_left.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToTopCorrectAmountOfData) {
    // data that was sent to left kernels
    auto data_left = getDataFromExternalChannel(0);

    EXPECT_EQ(data_left.size(), 0);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToBottomCorrectAmountOfData) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(2);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_top.size(), number_values);
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToRightCorrect) {
    // data that was sent to top kernels
    auto data_left = getDataFromExternalChannel(1);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_left.size(), number_values);
    if (data_left.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every row of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every column of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[j + i * BLOCK_SIZE] - data_left[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

TEST_F(LinpackKernelCommunicationTestLU, LUBlockExternalChannelOutputToBottomCorrect) {
    // data that was sent to top kernels
    auto data_top = getDataFromExternalChannel(2);

    size_t number_values = 0;
    for (int i = 0; i < BLOCK_SIZE; i++ ) {
        number_values += (BLOCK_SIZE - (i / CHUNK) * CHUNK);
    }
    EXPECT_EQ(data_top.size(), number_values);
    if (data_top.size() == number_values) {

        HOST_DATA_TYPE total_error = 0.0;

        size_t offset = 0;
        // for every column of a block
        for (int i = 0; i < BLOCK_SIZE; i++ ) {
            // for every row of a block
            for (int j = (i / CHUNK) * CHUNK; j < BLOCK_SIZE; j++) {
                total_error += std::abs(data->A[i + j * BLOCK_SIZE] - data_top[offset + (j - (i / CHUNK) * CHUNK)]);
            }
            offset += BLOCK_SIZE - (i / CHUNK) * CHUNK;
        }
        EXPECT_FLOAT_EQ(total_error, 0.0);
    }
}

// TODO implement tests for other kernels
