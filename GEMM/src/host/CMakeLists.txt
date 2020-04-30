
include_directories(../../../extern/cxxopts/include ../../../shared/)
include_directories(${IntelFPGAOpenCL_INCLUDE_DIRS})
include_directories(${CMAKE_BINARY_DIR}/src/common)

set(HOST_SOURCE execution_cannon.cpp main.cpp ../../../shared/setup/fpga_setup.cpp gemm_functionality.cpp)

find_package(BLAS)

add_executable(fgemm ${HOST_SOURCE})
target_link_libraries(fgemm ${IntelFPGAOpenCL_LIBRARIES})

if (BLAS_FOUND)
    target_compile_definitions(fgemm PRIVATE -D_USE_BLAS_)
    target_link_libraries(fgemm ${BLAS_LIBRARIES})
    target_link_options(fgemm PRIVATE ${BLAS_LINKER_FLAGS})
else()
    message(WARNING "No BLAS Library found. Slow reference implementation will be used for verification!")
endif()
