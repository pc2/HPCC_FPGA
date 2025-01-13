/*
Copyright (c) 2021 Marius Meyer

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
#ifndef HPCC_BASE_COMMUNICATION_TYPES_H_
#define HPCC_BASE_COMMUNICATION_TYPES_H_

#define DEFAULT_COMM_TYPE "AUTO"

#include <map>

namespace hpcc_base {

/**
 * @brief This enumeration contains all available communication types. They differ in the way data is exchanged between FPGAs. A special case is cpu_only which can be used to implement CPU references
 * 
 */
typedef enum _CommunicationType {

    /**
     * @brief Communication using the external channels extension
     * 
     */
    intel_external_channels,

    /**
     * @brief Copy the data from FPGA to CPU and send it via MPI
     * 
     */
    pcie_mpi,

    /**
     * @brief Communication using ACCL 
     */ 
    accl,

    /**
     * @brief Calculate the benchmark on CPU instead of FPGA
     * 
     */
    cpu_only,

    /**
     * @brief Use a pure UDP stack from the VNx project for communication
     */
    udp,

    /**
     * @brief Use the Aurora HLS library for communication
     */
    aurora,

    /**
     * @brief Indicates, that the use of the communication type is disabled
     * 
     */
    unsupported,

    /**
     * @brief Automatically detect communication type from kernel file name
     * 
     */
    automatic

} CommunicationType;

static const std::map<const std::string, CommunicationType> comm_to_str_map{ 
    {"IEC", CommunicationType::intel_external_channels}, 
    {"PCIE", CommunicationType::pcie_mpi},
    {"ACCL", CommunicationType::accl},
    {"CPU", CommunicationType::cpu_only},
    {"UNSUPPORTED", CommunicationType::unsupported},
    {"UDP", CommunicationType::udp},
    {"AURORA", CommunicationType::aurora},
    {"AUTO", CommunicationType::automatic}
    };
    
/**
 * @brief Serializes a enum of type CommunicationType into a string. The resulting string can be used with the function retrieveCommunicationType to get back the enum.
 * 
 * @param e the communication type that should be converted into a string
 * @return std::string String representation of the communication type
 */
static std::string commToString(CommunicationType c) {
    for (auto& entry : comm_to_str_map) {
        if (entry.second == c) {
            return entry.first;
        }
    }
    throw std::runtime_error("Communication type could not be converted to string!");
}

/**
 * @brief Deserializes a string into a enum of type CommunicationType. If the execution type is auto, the given kernel file name is used to determine the communication type. If this is not possible, an exception is thrown
 * 
 * @param exe_name String serialization of the communication tpye
 * @param kernel_filename the name of the used bitstream file
 * @return CommunicationType the determined communication type. Will throw a runtime error if it is not possible to retrieve the execution type
 */
static CommunicationType retrieveCommunicationType(std::string comm_name, std::string kernel_filename) {
    auto result = comm_to_str_map.find(comm_name);
    if (result != comm_to_str_map.end()) {
        if (result->second == CommunicationType::automatic) {
            for (auto &comm_type: comm_to_str_map) {
                if (kernel_filename.find(comm_type.first) != std::string::npos) {
                    return comm_type.second;
                }
            }
            throw std::runtime_error("Communication type could not be autodetected from kernel_filename: " + kernel_filename);
        } else {
            return result->second;
        }
    }
    throw std::runtime_error("Communication type could not be converted from string: " + comm_name);
}
}

#endif
