cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(FetchContent)

# ------------------------------------------------------------------------------
# A unit testing suite for C++
FetchContent_Declare(
  extern_googletest

  URL      https://github.com/google/googletest/archive/v1.10.x.tar.gz
  URL_HASH SHA256=4d7cd95cdb0ef420eed163696a906cccd70964801ea611b2020f31177432c27d)

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
  GIT_TAG             v1.2.1)

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
