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

#ifdef XILINX_FPGA
template <typename T>
struct aligned_allocator {

   //    typedefs
          typedef T value_type;
          typedef value_type* pointer;
          typedef const value_type* const_pointer;

	   pointer allocate(size_t pCount, const_pointer = 0){ 
	    	T* mem = 0;
	    	if (posix_memalign(reinterpret_cast<void**>(&mem), 1024 , sizeof(T) * pCount) != 0) {
	    		throw std::bad_alloc();
	        }
		return mem; 
	   }

	   void deallocate(pointer pPtr, size_t pCount) { 
	       free(pPtr);
	   }
};
	   
namespace cl {
    template <class T> using vector = std::vector<T,aligned_allocator<T>>; 
}
#endif

/**
 * @brief Contains all classes and methods needed by the Network benchmark
 * 
 */
namespace network {

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

    /**
     * @brief The data structure used to store all measurement results
     * 
     */
    typedef std::map<int, std::shared_ptr<std::vector<std::shared_ptr<ExecutionTimings>>>> CollectedResultMap;

/**
 * @brief The Network benchmark specific program settings
 * 
 */
class NetworkProgramSettings : public hpcc_base::BaseSettings {

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
class NetworkData {

public:

    /**
     * @brief Data type that contains all information needed for the execution of the kernel with a given data size.
     *          The class contains the data size, length of the inner loop and a buffer that takes the validatin data.
     *          In other workds it contains the information for a single run of the benchmark.
     * 
     */
    class NetworkDataItem {

    public:
        /**
         * @brief The used message size for the run in log2.
         * 
         */
        unsigned int messageSize;

        /**
         * @brief The loop length of the run (number of reptitions within the kernel). This can be used to extend the total execution time.
         * 
         */
        unsigned int loopLength;

        /**
         * @brief Data buffer that is used by the kernel to store received data. It can be used for validation by the host.
         * 
         */
        cl::vector<HOST_DATA_TYPE> validationBuffer;

        /**
         * @brief Construct a new Network Data Item object
         * 
         * @param messageSize The message size in bytes
         * @param loopLength The number of repetitions in the kernel
         */
        NetworkDataItem(unsigned int messageSize, unsigned int loopLength);
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
     * @param offset The used offset to scale the loop length. The higher the offset, the later the loop lenght will be decreased
     * @param decrease Number of steps the looplength will be decreased to the minimum
     */
    NetworkData(unsigned int max_looplength, unsigned int min_looplength,  unsigned int min_messagesize, unsigned int max_messagesize, unsigned int offset, unsigned int decrease);

};

/**
 * @brief Measured execution timing from the kernel execution
 * 
 */
class NetworkExecutionTimings {
public:

    /**
     * @brief A vector containing the timings for all repetitions for the kernel execution
     * 
     */
    CollectedResultMap timings;

};

/**
 * @brief Implementation of the Network benchmark
 * 
 */
class NetworkBenchmark : 
#ifdef USE_OCL_HOST
    public hpcc_base::HpccFpgaBenchmark<NetworkProgramSettings, cl::Device, cl::Context, cl::Program, NetworkData, NetworkExecutionTimings> 
#endif
#ifdef USE_XRT_HOST
    public hpcc_base::HpccFpgaBenchmark<NetworkProgramSettings, xrt::device, bool, xrt::uuid, NetworkData, NetworkExecutionTimings> 

#endif
   {
    protected:

    /**
     * @brief Additional input parameters of the Network benchmark
     * 
     * @param options 
     */
    void
    addAdditionalParseOptions(cxxopts::Options &options) override;

public:

    /**
     * @brief Network specific implementation of the data generation
     * 
     * @return std::unique_ptr<NetworkData> The input and output data of the benchmark
     */
    std::unique_ptr<NetworkData>
    generateInputData() override;

    /**
     * @brief Network specific implementation of the kernel execution
     * 
     * @param data The input and output data of the benchmark
     * @return std::unique_ptr<NetworkExecutionTimings> Measured runtimes of the kernel execution
     */
    std::unique_ptr<NetworkExecutionTimings>
    executeKernel(NetworkData &data) override;

    /**
     * @brief Network specific implementation of the execution validation
     * 
     * @param data The input and output data of the benchmark
     * @return true always, since no checks are done
     */
    bool
    validateOutputAndPrintError(NetworkData &data) override;

    /**
     * @brief Network specific implementation of printing the execution results
     * 
     * @param output Measured runtimes of the kernel execution
     */
    void
    collectAndPrintResults(const NetworkExecutionTimings &output) override;

    /**
     * @brief Construct a new Network Benchmark object. This construtor will directly setup
     *          The benchmark suing the given input parameters and the setupBenchmark() method
     * 
     * @param argc the number of program input parameters
     * @param argv the program input parameters as array of strings
     */
    NetworkBenchmark(int argc, char* argv[]);

    /**
     * @brief Construct a new Network Benchmark object
     */
    NetworkBenchmark();

};

} // namespace network


#endif // SRC_HOST_STREAM_BENCHMARK_H_
