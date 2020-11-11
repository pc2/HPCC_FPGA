# This file contains the default configuration for the Nallatech 520N board
# for the use with half precision floating point values.
# To use this configuration file, call cmake with the parameter
#
#     cmake [...] -DHPCC_FPGA_CONFIG="path to this file"
#


set(USE_MPI No CACHE BOOL "" FORCE)
set(USE_SVM No CACHE BOOL "" FORCE)
set(USE_HBM No CACHE BOOL "" FORCE)
set(FPGA_BOARD_NAME "p520_hpc_sg280l" CACHE STRING "" FORCE)
set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "" FORCE)

# STREAM specific options
# Defaults to a total of ~12GB data
set(DEFAULT_ARRAY_LENGTH 2147483648 CACHE STRING "" FORCE)
set(VECTOR_COUNT 16 CACHE STRING "" FORCE)
set(GLOBAL_MEM_UNROLL 2 CACHE STRING "" FORCE)
set(NUM_REPLICATIONS 4 CACHE STRING "" FORCE)
set(DEVICE_BUFFER_SIZE 32768 CACHE STRING "" FORCE)

# Set the data type since optional vector types are used
set(DATA_TYPE half CACHE STRING "" FORCE)
set(HOST_DATA_TYPE cl_${DATA_TYPE} CACHE STRING "" FORCE)
if (VECTOR_COUNT GREATER 1)
    set(DEVICE_DATA_TYPE ${DATA_TYPE}${VECTOR_COUNT} CACHE STRING "" FORCE)
else()
    set(DEVICE_DATA_TYPE ${DATA_TYPE} CACHE STRING "" FORCE)
endif()
