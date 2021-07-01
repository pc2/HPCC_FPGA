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
#ifndef SRC_HOST_EXECUTION_TPYES_H_
#define SRC_HOST_EXECUTION_TPYES_H_

namespace transpose {
namespace fpga_execution {

/**
 * @brief This enumeration contains all available communication types.
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
    pcie_mpi

} CommunicationType;

static const std::map<const std::string, CommunicationType> comm_to_str_map{ 
    {"IEC", CommunicationType::intel_external_channels}, 
    {"PCIE", CommunicationType::pcie_mpi}
    };

static std::string commToString(CommunicationType c) {
    for (auto& entry : comm_to_str_map) {
        if (entry.second == c) {
            return entry.first;
        }
    }
    throw new std::runtime_error("Communication type could not be converted to string!");
}

static CommunicationType stringToComm(std::string comm_name) {
    auto result = comm_to_str_map.find(comm_name);
    if (result != comm_to_str_map.end()) {
        return result->second;
    }
    throw new std::runtime_error("Communication type could not be converted from string: " + comm_name);
}

}
}

#endif