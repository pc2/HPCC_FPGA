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
#ifndef SRC_HOST_DATA_HANDLER_TPYES_H_
#define SRC_HOST_DATA_HANDLER_TPYES_H_

/* C++ standard library headers */
#include <map>

namespace transpose {
namespace data_handler {

/**
 * @brief This enumeration contains all available data handler types.
 * 
 */
typedef enum _DataHandlerType {

    /**
     * @brief The matrix is already blockwise diagonally distributed which only required data exchange with a single node
     * 
     */
    diagonal,

    /**
     * @brief Classical distribution of the matrix in a PQ grid
     * 
     */
    pq


} DataHandlerType;

static const std::map<const std::string, DataHandlerType> comm_to_str_map{ 
    {"DIAG", DataHandlerType::diagonal}, 
    {"PQ", DataHandlerType::pq},
    };

static std::string handlerToString(DataHandlerType c) {
    for (auto& entry : comm_to_str_map) {
        if (entry.second == c) {
            return entry.first;
        }
    }
    throw new std::runtime_error("Communication type could not be converted to string!");
}

static DataHandlerType stringToHandler(std::string comm_name) {
    auto result = comm_to_str_map.find(comm_name);
    if (result != comm_to_str_map.end()) {
        return result->second;
    }
    throw new std::runtime_error("Communication type could not be converted from string: " + comm_name);
}

}
}

#endif