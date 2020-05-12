
#ifndef SRC_HOST_PROGRAM_SETTINGS_H_
#define SRC_HOST_PROGRAM_SETTINGS_H_

#include "parameters.h"

/* C++ standard library headers */
#include <memory>

#include "CL/opencl.h"

/*
Short description of the program.
Moreover the version and build time is also compiled into the description.
*/

/*
Short description of the program
*/
#define PROGRAM_DESCRIPTION "Implementation of the LINPACK benchmark"\
                            " proposed in the HPCC benchmark suite for FPGA.\n"\
                            "Version: " VERSION "\n"


struct ProgramSettings {
    uint numRepetitions;
    uint matrixSize;
    int defaultPlatform;
    int defaultDevice;
    std::string kernelFileName;
};


#endif
