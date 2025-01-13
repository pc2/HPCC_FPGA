cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(FetchContent)

# ------------------------------------------------------------------------------
# A unit testing suite for C++
FetchContent_Declare(
  extern_googletest

  URL      https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
  URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5)

FetchContent_GetProperties(extern_googletest)
if(NOT extern_googletest_POPULATED)
  message(STATUS "Fetching mandatory build dependency GoogleTest")
  FetchContent_Populate(extern_googletest)
  add_subdirectory(
    ${extern_googletest_SOURCE_DIR} 
    ${extern_googletest_BINARY_DIR} 
    EXCLUDE_FROM_ALL)
endif()

# ------------------------------------------------------------------------------
# A library that provides helper classes for the development of HLS codes.
# This benchmark suite uses only the CMake files to find the Intel and Vitis installations
FetchContent_Declare(
  extern_hlslib

  # unfortunately they do not use releases, so the latest commit was used
  GIT_REPOSITORY      https://github.com/definelicht/hlslib.git
  GIT_TAG             master)

FetchContent_GetProperties(extern_hlslib)
if(NOT extern_hlslib_POPULATED)
  message(STATUS "Fetching mandatory build dependency hlslib")
  FetchContent_Populate(extern_hlslib)
  set(extern_hlslib_SOURCE_DIR ${extern_hlslib_SOURCE_DIR} PARENT_SCOPE)
endif()

# ------------------------------------------------------------------------------
# A header only library to parse program inputs
FetchContent_Declare(
  extern_cxxopts

  URL      https://github.com/jarro2783/cxxopts/archive/v2.2.1.tar.gz
  URL_HASH SHA256=984aa3c8917d649b14d7f6277104ce38dd142ce378a9198ec926f03302399681)

FetchContent_GetProperties(extern_cxxopts)
if(NOT extern_cxxopts_POPULATED)
  message(STATUS "Fetching mandatory build dependency cxxopts")
  FetchContent_Populate(extern_cxxopts)
  add_subdirectory(
    ${extern_cxxopts_SOURCE_DIR}
    ${extern_cxxopts_BINARY_DIR}
    EXCLUDE_FROM_ALL)
endif()

if (USE_ACCL)
message(STATUS "ACCL was selected. Fetch ACCL dependencies")
# -------------------------------------------------------------------------------
# ACCL Library
FetchContent_Declare(
	extern_accl

    GIT_REPOSITORY	https://github.com/Xilinx/ACCL.git
	GIT_TAG		dev)

FetchContent_GetProperties(extern_accl)
if(NOT extern_accl_POPULATED)
	message(STATUS "Fetching mandatory build dependency ACCL")
	FetchContent_Populate(extern_accl)
	set(extern_accl_SOURCE_DIR ${extern_accl_SOURCE_DIR} PARENT_SCOPE)
endif()

# -------------------------------------------------------------------------------
# TCP Library
FetchContent_Declare(
	extern_accl_tcp

    GIT_REPOSITORY	https://github.com/fpgasystems/Vitis_with_100Gbps_TCP-IP.git
	  GIT_TAG		vitis_2022_1)

FetchContent_GetProperties(extern_accl_tcp)
if(NOT extern_accl_tcp_POPULATED)
	message(STATUS "Fetching mandatory build dependency ACCL TCP stack")
	FetchContent_Populate(extern_accl_tcp)
	set(extern_accl_tcp_SOURCE_DIR ${extern_accl_tcp_SOURCE_DIR} PARENT_SCOPE)
endif()
endif()

if (USE_ACCL OR USE_UDP)
# -------------------------------------------------------------------------------
# UDP Library
FetchContent_Declare(
	extern_vnx_udp

    GIT_REPOSITORY	https://github.com/Xilinx/xup_vitis_network_example.git
	  GIT_TAG		master)

FetchContent_GetProperties(extern_vnx_udp)
if(NOT extern_vnx_udp_POPULATED)
	message(STATUS "Fetching mandatory build dependency VNx UDP stack")
	FetchContent_Populate(extern_vnx_udp)
	set(extern_vnx_udp_SOURCE_DIR ${extern_vnx_udp_SOURCE_DIR} PARENT_SCOPE)
endif()
endif()
# ------------------------------------------------------------------------------
# A header only library for handling json
FetchContent_Declare(
  extern_json

  URL      https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
  URL_HASH SHA256=8c4b26bf4b422252e13f332bc5e388ec0ab5c3443d24399acb675e68278d341f)

FetchContent_GetProperties(extern_json)
if(NOT extern_json_POPULATED)
  message(STATUS "Fetching mandatory build dependency json")
  FetchContent_Populate(extern_json)
  add_subdirectory(
    ${extern_json_SOURCE_DIR}
    ${extern_json_BINARY_DIR}
    EXCLUDE_FROM_ALL)
  set(extern_json_SOURCE_DIR ${extern_json_SOURCE_DIR} PARENT_SCOPE)
endif()


