
#ifndef SRC_HOST_PROGRAM_SETTINGS_H_
#define SRC_HOST_PROGRAM_SETTINGS_H_

#include "parameters.h"

/* C++ standard library headers */
#include <memory>

#include "CL/opencl.h"



#define PROGRAM_DESCRIPTION "Implementation of the STREAM benchmark"\
                            " proposed in the HPCC benchmark suite for FPGA.\n"\
                            "Version: " VERSION "\n"


/**
* A struct that is used to store the porgram settings
* provided by command line arguments.
*/
struct ProgramSettings {
    uint numRepetitions;
    uint streamArraySize;
    uint kernelReplications;
    bool useMemoryInterleaving;
    int defaultPlatform;
    int defaultDevice;
    std::string kernelFileName;
    bool useSingleKernel;
};


#endif
