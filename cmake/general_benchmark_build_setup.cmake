cmake_policy(VERSION 3.13)
INCLUDE (CheckTypeSize)

set (CMAKE_CXX_STANDARD 11)

# Download build dependencies
add_subdirectory(${CMAKE_SOURCE_DIR}/../extern ${CMAKE_BINARY_DIR}/extern)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
endif()

# Host code specific options
set(DEFAULT_REPETITIONS 10 CACHE STRING "Default number of repetitions")
set(DEFAULT_DEVICE -1 CACHE STRING "Index of the default device to use")
set(DEFAULT_PLATFORM -1 CACHE STRING "Index of the default platform to use")
set(USE_OPENMP ${USE_OPENMP} CACHE BOOL "Use OpenMP in the host code")
set(USE_MPI ${USE_MPI} CACHE BOOL "Compile the host code with MPI support. This has to be supported by the host code.")
set(USE_SVM No CACHE BOOL "Use SVM pointers instead of creating buffers on the board and transferring the data there before execution.")
set(USE_HBM No CACHE BOOL "Use host code specific to HBM FPGAs")
set(USE_CUSTOM_KERNEL_TARGETS No CACHE BOOL "Enable build targets for custom kernels")

mark_as_advanced(USE_MPI)

if (USE_SVM AND USE_HBM)
    message(ERROR "Misconfiguration: Can not use USE_HBM and USE_SVM at the same time because they target different memory architectures")
endif()

# Set the used data type
if (NOT DATA_TYPE)
    set(DATA_TYPE float CACHE STRING "Data type used for calculation")
else()
    set(DATA_TYPE ${DATA_TYPE} CACHE STRING "Data type used for calculation")
endif()

if (NOT HOST_DATA_TYPE OR NOT DEVICE_DATA_TYPE)
    set(HOST_DATA_TYPE cl_${DATA_TYPE})
    set(DEVICE_DATA_TYPE ${DATA_TYPE})
endif()

# Setup CMake environment
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${extern_hlslib_SOURCE_DIR}/cmake)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Search for general dependencies
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()
if (USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-D_USE_MPI_)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    link_libraries(${MPI_LIBRARIES})
endif()
find_package(IntelFPGAOpenCL)
find_package(Vitis)
find_package(Python3)

if (NOT VITIS_FOUND AND NOT INTELFPGAOPENCL_FOUND)
    message(ERROR "Xilinx Vitis or Intel FPGA OpenCL SDK required!")
endif()

if (NOT Python3_Interpreter_FOUND)
    message(WARNING "Python 3 interpreter could not be found! It might be necessary to generate the final kernel source code!")
endif()

# Set the size of the used data type
list(APPEND CMAKE_REQUIRED_INCLUDES ${IntelFPGAOpenCL_INCLUDE_DIRS} ${Vitis_INCLUDE_DIRS})
list(APPEND CMAKE_EXTRA_INCLUDE_FILES "CL/opencl.h")
check_type_size("${HOST_DATA_TYPE}" DATA_TYPE_SIZE)

# Configure the header file with definitions used by the host code
configure_file(
        "${CMAKE_SOURCE_DIR}/src/common/parameters.h.in"
        "${CMAKE_BINARY_DIR}/src/common/parameters.h"
)
include_directories(${CMAKE_BINARY_DIR}/src/common)


# Find Xilinx settings files
file(GLOB POSSIBLE_XILINX_LINK_FILES ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx*.ini)
file(GLOB POSSIBLE_XILINX_LINK_GENERATOR_FILES ${CMAKE_SOURCE_DIR}/settings/settings.link.xilinx*.generator.ini)
file(GLOB POSSIBLE_XILINX_COMPILE_FILES ${CMAKE_SOURCE_DIR}/settings/settings.compile.xilinx*.ini)
list(REMOVE_ITEM POSSIBLE_XILINX_LINK_FILES EXCLUDE REGEX ${POSSIBLE_XILINX_LINK_GENERATOR_FILES})

if (POSSIBLE_XILINX_LINK_FILES)
    list(GET POSSIBLE_XILINX_LINK_FILES 0 POSSIBLE_XILINX_LINK_FILES)
endif()

if (POSSIBLE_XILINX_COMPILE_FILES)
    list(GET POSSIBLE_XILINX_COMPILE_FILES 0 POSSIBLE_XILINX_COMPILE_FILES)
endif()

if (POSSIBLE_XILINX_LINK_GENERATOR_FILES)
list(GET POSSIBLE_XILINX_LINK_GENERATOR_FILES 0 POSSIBLE_XILINX_LINK_GENERATOR_FILES)
endif()

if (XILINX_LINK_SETTINGS_FILE AND XILINX_LINK_SETTINGS_FILE MATCHES ".*.generator.ini$")
	set(POSSIBLE_XILINX_LINK_GENERATOR_FILES ${XILINX_LINK_SETTINGS_FILE})
endif()
if (XILINX_LINK_SETTINGS_FILE AND NOT XILINX_LINK_SETTINGS_FILE MATCHES ".*.generator.ini$")
	set(POSSIBLE_XILINX_LINK_FILES ${XILINX_LINK_SETTINGS_FILE})
	set(POSSIBLE_XILINX_LINK_GENERATOR_FILES "")
endif()

if (POSSIBLE_XILINX_LINK_GENERATOR_FILES)
    set(XILINX_GENERATE_LINK_SETTINGS Yes)
    set(POSSIBLE_XILINX_LINK_FILES ${POSSIBLE_XILINX_LINK_GENERATOR_FILES})
else()
    set(XILINX_GENERATE_LINK_SETTINGS No)
endif()

# Set FPGA board name if specified in the environment
if (NOT DEFINED FPGA_BOARD_NAME)
    if (DEFINED ENV{FPGA_BOARD_NAME})
        set(FPGA_BOARD_NAME $ENV{FPGA_BOARD_NAME})
    else()
        set(FPGA_BOARD_NAME p520_hpc_sg280l)
    endif()
endif()


# FPGA specific options
set(FPGA_BOARD_NAME ${FPGA_BOARD_NAME} CACHE STRING "Name of the target FPGA board")
if (VITIS_FOUND)
    set(XILINX_GENERATE_LINK_SETTINGS ${XILINX_GENERATE_LINK_SETTINGS} CACHE BOOL "Generate the link settings depending on the number of replications for the kernels")
    set(XILINX_COMPILE_FLAGS "-j 40" CACHE STRING "Additional compile flags for the v++ compiler")
    set(XILINX_LINK_SETTINGS_FILE ${POSSIBLE_XILINX_LINK_FILES} CACHE FILEPATH "The link settings file that should be used when generating is disabled")
    set(XILINX_COMPILE_SETTINGS_FILE ${POSSIBLE_XILINX_COMPILE_FILES} CACHE FILEPATH "Compile settings file for the v++ compiler")
    separate_arguments(XILINX_COMPILE_FLAGS)
endif()
if (INTELFPGAOPENCL_FOUND)
    set(AOC_FLAGS "-fpc -fp-relaxed -no-interleaving=default" CACHE STRING "Used flags for the AOC compiler")
    set(INTEL_CODE_GENERATION_SETTINGS "" CACHE FILEPATH "Code generation settings file for the intel targets")
    separate_arguments(AOC_FLAGS)
endif()

set(CODE_GENERATOR "${CMAKE_SOURCE_DIR}/../scripts/code_generator/generator.py" CACHE FILEPATH "Path to the code generator executable")
set(CUSTOM_KERNEL_FOLDER ${CMAKE_SOURCE_DIR}/src/device/custom/)


set(kernel_emulation_targets_intel "" CACHE INTERNAL "")
set(kernel_emulation_targets_xilinx "" CACHE INTERNAL "")

# Add subdirectories of the project
add_subdirectory(${CMAKE_SOURCE_DIR}/src/device)

if (USE_CUSTOM_KERNEL_TARGETS)
    message(STATUS "Create custom kernel targets.")
    add_subdirectory(${CUSTOM_KERNEL_FOLDER})
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/src/host)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif()
