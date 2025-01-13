/*
Copyright (c) 2019 Marius Meyer

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

#ifndef SRC_HOST_NETWORK_BENCHMARK_H_
#define SRC_HOST_NETWORK_BENCHMARK_H_

/* C++ standard library headers */
#include <complex>
#include <memory>

/* Project's headers */
#include "hpcc_benchmark.hpp"
#include "parameters.h"

#ifdef USE_DEPRECATED_HPP_HEADER
template <typename T>
struct aligned_allocator {

    //    typedefs
    typedef T value_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;

    pointer allocate(size_t pCount, const_pointer = 0)
    {
        T *mem = 0;
        if (posix_memalign(reinterpret_cast<void **>(&mem), 4096, sizeof(T) * pCount) != 0) {
            throw std::bad_alloc();
        }
        return mem;
    }

    void deallocate(pointer pPtr, size_t pCount) { free(pPtr); }
};

namespace cl
{
template <class T>
using vector = std::vector<T, aligned_allocator<T>>;
}
#endif

/**
 * @brief Contains all classes and methods needed by the Network benchmark
 *
 */
namespace network
{

/**
 * @brief This data struct is part of the CollectedResultMap.
 *         It is used to store the measurement results for a single rank
 *          executed with a specific loop length and message size
 *
 */
struct ExecutionTimings {

    /**
     * @brief The number of messages that were sent for this measurement
     *
     */
    cl_uint looplength;

    /**
     * @brief The size of the messages in bytes in log2
     *
     */
    cl_uint messageSize;

    /**
     * @brief The kernel runtimes for each repetition in seconds
     *
     */
    std::vector<double> calculationTimings;
};

struct ExecutionResult {
    std::vector<ExecutionTimings> execution_timings;
    /**
     * @brief maximum of minimum calculation time, filled by collectResults
     *
     */
    double maxMinCalculationTime;

    /**
     * @brief maximum of calculated bandwidths, filled by collectResults
     *
     */
    double maxCalcBW;
};

/**
 * @brief The data structure used to store all measurement results
 *
 */
typedef std::map<int, ExecutionResult> CollectedTimingsMap;

/**
 * @brief The Network benchmark specific program settings
 *
 */
class NetworkProgramSettings : public hpcc_base::BaseSettings
{

  public:
    /**
     * @brief Initial number of sent messages per message size
     *
     */
    uint maxLoopLength;

    /**
     * @brief Minimum number of sent messages per message size
     *
     */
    uint minLoopLength;

    /**
     * @brief Log2 of maximum message size
     *
     */
    uint maxMessageSize;

    /**
     * @brief Log2 of minimum message size
     *
     */
    uint minMessageSize;

    /**
     * @brief Step size for tested message sizes
     *
     */
    uint stepSize;

    /**
     * @brief Offset that is used before the loop length will be reduced for higher message sizes
     *
     */
    uint llOffset;

    /**
     * @brief Number of steps the loop length is decreased after the offset is reached
     *
     */
    uint llDecrease;

    /**
     * @brief Use the second command kernel to schedule sends and receives directly from PL
     *
     */
    bool accl_from_programable_logic;

    /**
     * @brief Forward data to AXI stream instead of global memory to further reduce latency
     */
    bool accl_axi_stream;

    /**
     * @brief his is automatically set to true if one of pcie_reverse_write_pcie, pcie_reverse_read_pcie,
     * or pcie_reverse_execute_kernel is set to true. The reverse PCIe experiment will be executed in that case.
     *
     */
    bool pcie_reverse;

    /**
     * @brief If true, the benchmark will execute the reverse PCIe benchmark instead. It will write data to the FPGA.
     * The other pcie_reverse flags can be set to do additional operations within the measurement.
     *
     */
    bool pcie_reverse_write_pcie;

    /**
     * @brief If true, the benchmark will execute the reverse PCIe benchmark instead. It will execute an empty kernel.
     * The other pcie_reverse flags can be set to do additional operations within the measurement.
     *
     */
    bool pcie_reverse_execute_kernel;

    /**
     * @brief If true, the benchmark will execute the reverse PCIe benchmark instead. It will read data from the FPGA.
     * The other pcie_reverse flags can be set to do additional operations within the measurement.
     *
     */
    bool pcie_reverse_read_pcie;

    /**
     * @brief If true, the reverse experiments are executed in batch mode per looplength to make use of the scheduling
     * queues
     *
     */
    bool pcie_reverse_batch;

    /**
     * @brief Log2 of the maximum packet payload used in chunks of 64 Bytes. 6 -> 2^6 = 4 KiB
     */
    uint payload_size;

    /**
     * @brief Construct a new Network Program Settings object
     *
     * @param results the result map from parsing the program input parameters
     */
    NetworkProgramSettings(cxxopts::ParseResult &results);

    /**
     * @brief Get a map of the settings. This map will be used to print the final configuration.
     *
     * @return a map of program parameters. keys are the name of the parameter.
     */
    std::map<std::string, std::string> getSettingsMap() override;
};

/**
 * @brief Data class for the network benchmark
 *
 */
class NetworkData
{

  public:
    /**
     * @brief Data type that contains all information needed for the execution of the kernel with a given data size.
     *          The class contains the data size, length of the inner loop and a buffer that takes the validatin data.
     *          In other workds it contains the information for a single run of the benchmark.
     *
     */
    class NetworkDataItem
    {

      public:
        /**
         * @brief The used message size for the run in log2.
         *
         */
        unsigned int messageSize;

        /**
         * @brief The loop length of the run (number of reptitions within the kernel). This can be used to extend the
         * total execution time.
         *
         */
        unsigned int loopLength;

        /**
         * @brief Data buffer that is used by the kernel to store received data. It can be used for validation by the
         * host.
         *
         */
        cl::vector<HOST_DATA_TYPE> validationBuffer;

        /**
         * @brief Construct a new Network Data Item object
         *
         * @param messageSize The message size in bytes
         * @param loopLength The number of repetitions in the kernel
         * @param replications The number of kernel replications
         */
        NetworkDataItem(unsigned int messageSize, unsigned int loopLength, unsigned int replications);
    };

    /**
     * @brief Data items that are used for the benchmark.
     *
     */
    std::vector<NetworkDataItem> items;

    /**
     * @brief Construct a new Network Data object
     *
     * @param max_looplength The maximum number of iterations that should be done for a message size
     * @param min_looplength The minimum number of iterations that should be done for a message size
     * @param max_messagesize The minimum message size
     * @param max_messagesize The maximum message size
     * @param stepSize Step size used to generate tested message sizes
     * @param offset The used offset to scale the loop length. The higher the offset, the later the loop lenght will be
     * decreased
     * @param decrease Number of steps the looplength will be decreased to the minimum
     * @param replications The number of kernel replications
     */
    NetworkData(unsigned int max_looplength, unsigned int min_looplength, unsigned int min_messagesize,
                unsigned int max_messagesize, unsigned int stepSize, unsigned int offset, unsigned int decrease,
                unsigned int replications);
};

/**
 * @brief Implementation of the Network benchmark
 *
 */
class NetworkBenchmark :
#ifdef USE_OCL_HOST
    public hpcc_base::HpccFpgaBenchmark<network::NetworkProgramSettings, cl::Device, cl::Context, cl::Program,
                                        network::NetworkData>
#endif
#ifdef USE_XRT_HOST
#ifdef USE_ACCL
    public hpcc_base::HpccFpgaBenchmark<network::NetworkProgramSettings, xrt::device, fpga_setup::ACCLContext,
                                        xrt::uuid, network::NetworkData>
#else
    public hpcc_base::HpccFpgaBenchmark<network::NetworkProgramSettings, xrt::device, fpga_setup::VNXContext, xrt::uuid,
                                        network::NetworkData>
#endif
#endif
{
  protected:
    /**
     * @brief Data structure used to store the number of errors for each message size
     *
     */
    std::map<std::string, int> errors;

    /**
     * @brief Additional input parameters of the Network benchmark
     *
     * @param options
     */
    void addAdditionalParseOptions(cxxopts::Options &options) override;

  public:
    CollectedTimingsMap collected_timings;

    json getTimingsJson() override
    {
        json j;
        for (const auto &timing : collected_timings) {
            json timing_json;
            timing_json["maxMinCalculationTime"] = timing.second.maxMinCalculationTime;
            timing_json["maxCalcBW"] = timing.second.maxCalcBW;
            std::vector<json> timings_json;
            for (const auto &execution_timing : timing.second.execution_timings) {
                json single_timing_json;
                single_timing_json["looplength"] = execution_timing.looplength;
                single_timing_json["messageSize"] = execution_timing.messageSize;
                std::vector<json> calculation_timings;
                for (const auto &timing : execution_timing.calculationTimings) {
                    json j;
                    j["unit"] = "s";
                    j["value"] = timing;
                    calculation_timings.push_back(j);
                }
                single_timing_json["timings"] = calculation_timings;
                timings_json.push_back(single_timing_json);
            }
            timing_json["timings"] = timings_json;

            j[std::to_string(timing.first)] = timing_json;
        }
        return j;
    }

    /**
     * @brief Network specific implementation of the data generation
     *
     * @return std::unique_ptr<NetworkData> The input and output data of the benchmark
     */
    std::unique_ptr<NetworkData> generateInputData() override;

    /**
     * @brief Network specific implementation of the kernel execution
     *
     * @param data The input and output data of the benchmark
     * @return std::unique_ptr<NetworkExecutionTimings> Measured runtimes of the kernel execution
     */
    void executeKernel(network::NetworkData &data) override;

    /**
     * @brief Network specific implementation of the execution validation
     *
     * @param data The input and output data of the benchmark
     * @return true always, since no checks are done
     */
    bool validateOutput(network::NetworkData &data) override;

    /**
     * @brief Network specific implementation of the error printing
     *
     */
    void printError() override;

    /**
     * @brief Network specific implementation of collecting the execution results
     *
     * @param output Measured runtimes of the kernel execution
     */
    void collectResults() override;

    /**
     * @brief Network specifig implementation of the printing the execution results
     *
     */
    void printResults() override;

    /**
     * @brief Construct a new Network Benchmark object. This construtor will directly setup
     *          The benchmark suing the given input parameters and the setupBenchmark() method
     *
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    NetworkBenchmark(int argc, char *argv[]);

    /**
     * @brief Construct a new Network Benchmark object
     */
    NetworkBenchmark();
};

} // namespace network

#endif // SRC_HOST_STREAM_BENCHMARK_H_
