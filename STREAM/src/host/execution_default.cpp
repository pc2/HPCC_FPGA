/*
Copyright (c) 2019 Marius Meyer

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

/* Related header files */
#include "execution.hpp"

/* C++ standard library headers */
#include <memory>
#include <vector>
#include <chrono>

/* External library headers */
#include "CL/opencl.h"

#ifdef INTEL_FPGA
#include "CL/cl_ext_intelfpga.h"
#endif
/* Project's headers */

namespace bm_execution {

    void initialize_buffers(const hpcc_base::ExecutionSettings<stream::StreamProgramSettings> &config, unsigned int data_per_kernel,
                            std::vector<cl::Buffer> &Buffers_A, std::vector<cl::Buffer> &Buffers_B,
                            std::vector<cl::Buffer> &Buffers_C);

    bool initialize_queues_and_kernels(const hpcc_base::ExecutionSettings<stream::StreamProgramSettings> &config,
                                       unsigned int data_per_kernel, const std::vector<cl::Buffer> &Buffers_A,
                                       const std::vector<cl::Buffer> &Buffers_B,
                                       const std::vector<cl::Buffer> &Buffers_C,
                                       std::vector<cl::Kernel> &test_kernels, std::vector<cl::Kernel> &copy_kernels,
                                       std::vector<cl::Kernel> &scale_kernels, std::vector<cl::Kernel> &add_kernels,
                                       std::vector<cl::Kernel> &triad_kernels,
                                       std::vector<cl::CommandQueue> &command_queues);

    bool initialize_queues_and_kernels_single(const hpcc_base::ExecutionSettings<stream::StreamProgramSettings> &config,
                                       unsigned int data_per_kernel, const std::vector<cl::Buffer> &Buffers_A,
                                       const std::vector<cl::Buffer> &Buffers_B,
                                       const std::vector<cl::Buffer> &Buffers_C,
                                       std::vector<cl::Kernel> &test_kernels, std::vector<cl::Kernel> &copy_kernels,
                                       std::vector<cl::Kernel> &scale_kernels, std::vector<cl::Kernel> &add_kernels,
                                       std::vector<cl::Kernel> &triad_kernels,
                                       HOST_DATA_TYPE* A,
                                       HOST_DATA_TYPE* B,
                                       HOST_DATA_TYPE* C,
                                       std::vector<cl::CommandQueue> &command_queues);

/*
    Implementation for the single kernel.
     @copydoc bm_execution::calculate()
    */
    std::unique_ptr<stream::StreamExecutionTimings>
    calculate(const hpcc_base::ExecutionSettings<stream::StreamProgramSettings>& config,
            HOST_DATA_TYPE* A,
            HOST_DATA_TYPE* B,
            HOST_DATA_TYPE* C) {

        unsigned data_per_kernel = config.programSettings->streamArraySize/config.programSettings->kernelReplications;

        std::vector<cl::Buffer> Buffers_A;
        std::vector<cl::Buffer> Buffers_B;
        std::vector<cl::Buffer> Buffers_C;
        std::vector<cl::Kernel> test_kernels;
        std::vector<cl::Kernel> copy_kernels;
        std::vector<cl::Kernel> scale_kernels;
        std::vector<cl::Kernel> add_kernels;
        std::vector<cl::Kernel> triad_kernels;
        std::vector<cl::CommandQueue> command_queues;

        //
        // Setup buffers
        //
        initialize_buffers(config, data_per_kernel, Buffers_A, Buffers_B, Buffers_C);

        //
        // Setup kernels
        //
        bool success = false;
        if (config.programSettings->useSingleKernel) {
            success = initialize_queues_and_kernels_single(config, data_per_kernel, Buffers_A, Buffers_B, Buffers_C, test_kernels,
                                          copy_kernels, scale_kernels,
                                          add_kernels, triad_kernels, A, B, C, command_queues);
        }
        else {
            success = initialize_queues_and_kernels(config, data_per_kernel, Buffers_A, Buffers_B, Buffers_C, test_kernels,
                                          copy_kernels, scale_kernels,
                                          add_kernels, triad_kernels, command_queues);
        }
        if (!success) {
            return std::unique_ptr<stream::StreamExecutionTimings>(nullptr);
        }

        //
        // Setup counters for runtime measurement
        //
        std::map<std::string, std::vector<double>> timingMap;
        timingMap.insert({PCIE_READ_KEY, std::vector<double>()});
        timingMap.insert({PCIE_WRITE_KEY, std::vector<double>()});
        timingMap.insert({COPY_KEY, std::vector<double>()});
        timingMap.insert({SCALE_KEY, std::vector<double>()});
        timingMap.insert({ADD_KEY, std::vector<double>()});
        timingMap.insert({TRIAD_KEY, std::vector<double>()});

        //
        // Do first test execution
        //
        std::chrono::time_point<std::chrono::high_resolution_clock> startExecution, endExecution;
        std::chrono::duration<double> duration;
        // Time checking with test kernel
        for (int i=0; i<config.programSettings->kernelReplications; i++) {
#ifdef USE_SVM
            ASSERT_CL(clEnqueueSVMMap(command_queues[i](), CL_FALSE,
                                CL_MAP_READ | CL_MAP_WRITE,
                                reinterpret_cast<void *>(A),
                                sizeof(HOST_DATA_TYPE) * data_per_kernel, 0,
                                NULL, NULL));

#else
            ASSERT_CL(command_queues[i].enqueueWriteBuffer(Buffers_A[i], CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*data_per_kernel, &A[data_per_kernel*i]));
#endif
        }
        for (int i=0; i<config.programSettings->kernelReplications; i++) {
            ASSERT_CL(command_queues[i].finish());
        }
        startExecution = std::chrono::high_resolution_clock::now();
        for (int i=0; i<config.programSettings->kernelReplications; i++) {
            ASSERT_CL(command_queues[i].enqueueNDRangeKernel(test_kernels[i], cl::NullRange, cl::NDRange(1)));
        }
        for (int i=0; i<config.programSettings->kernelReplications; i++) {
            ASSERT_CL(command_queues[i].finish());
        }
        endExecution = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double>>
                (endExecution - startExecution);
        std::cout << "Each test below will take on the order of " << duration.count() * 1.0e6 << " microseconds." << std::endl;

        std::cout << HLINE;

        std::cout << "WARNING -- The above is only a rough guideline." << std::endl;
        std::cout << "For best results, please be sure you know the" << std::endl;
        std::cout << "precision of your system timer." << std::endl;
        std::cout << HLINE;

        for (int i=0; i<config.programSettings->kernelReplications; i++) {
#ifdef USE_SVM
            ASSERT_CL(clEnqueueSVMUnmap(command_queues[i](),
                        reinterpret_cast<void *>(A), 0,
                        NULL, NULL));

#else
            ASSERT_CL(command_queues[i].enqueueReadBuffer(Buffers_A[i], CL_FALSE, 0, sizeof(HOST_DATA_TYPE)*data_per_kernel, &A[data_per_kernel*i]));
#endif
        }
        for (int i=0; i<config.programSettings->kernelReplications; i++) {
            ASSERT_CL(command_queues[i].finish());
        }


        //
        // Do actual benchmark measurements
        //
        for (uint r = 0; r < config.programSettings->numRepetitions; r++) {


            startExecution = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
#ifdef USE_SVM
                clEnqueueSVMMap(command_queues[i](), CL_FALSE,
                            CL_MAP_READ | CL_MAP_WRITE,
                            reinterpret_cast<void *>(&A[data_per_kernel * i]),
                            sizeof(HOST_DATA_TYPE) * data_per_kernel, 0,
                            NULL, NULL);
                clEnqueueSVMMap(command_queues[i](), CL_FALSE,
                            CL_MAP_READ | CL_MAP_WRITE,
                            reinterpret_cast<void *>(&B[data_per_kernel * i]),
                            sizeof(HOST_DATA_TYPE) * data_per_kernel, 0,
                            NULL, NULL);
                clEnqueueSVMMap(command_queues[i](), CL_FALSE,
                            CL_MAP_READ | CL_MAP_WRITE,
                            reinterpret_cast<void *>(&C[data_per_kernel * i]),
                            sizeof(HOST_DATA_TYPE) * data_per_kernel, 0,
                            NULL, NULL);
#else
                command_queues[i].enqueueWriteBuffer(Buffers_A[i], CL_FALSE, 0,
                                                        sizeof(HOST_DATA_TYPE) * data_per_kernel,
                                                        &A[data_per_kernel * i]);
                command_queues[i].enqueueWriteBuffer(Buffers_B[i], CL_FALSE, 0,
                                                        sizeof(HOST_DATA_TYPE) * data_per_kernel,
                                                        &B[data_per_kernel * i]);
                command_queues[i].enqueueWriteBuffer(Buffers_C[i], CL_FALSE, 0,
                                                        sizeof(HOST_DATA_TYPE) * data_per_kernel,
                                                        &C[data_per_kernel * i]);
#endif
            }

            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                command_queues[i].finish();
            }


            endExecution = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>
                    (endExecution - startExecution);
            timingMap[PCIE_WRITE_KEY].push_back(duration.count());

            int err;
            cl::UserEvent copy_user_event(*config.context, &err);
            ASSERT_CL(err);
            std::vector<cl::Event> copy_start_events({copy_user_event});
            std::vector<cl::Event> copy_events(config.programSettings->kernelReplications);
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                command_queues[i].enqueueNDRangeKernel(copy_kernels[i], cl::NullRange, cl::NDRange(1), cl::NDRange(1), &copy_start_events, &copy_events[i]);
            }

            cl::UserEvent scale_user_event(*config.context, &err);
            ASSERT_CL(err);
            std::vector<cl::Event> scale_start_events({scale_user_event});
            std::vector<cl::Event> scale_events(config.programSettings->kernelReplications);
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                command_queues[i].enqueueNDRangeKernel(scale_kernels[i], cl::NullRange, cl::NDRange(1), cl::NDRange(1), &scale_start_events, &scale_events[i]);
            }

            cl::UserEvent add_user_event(*config.context, &err);
            ASSERT_CL(err);
            std::vector<cl::Event> add_start_events({add_user_event});
            std::vector<cl::Event> add_events(config.programSettings->kernelReplications);
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                command_queues[i].enqueueNDRangeKernel(add_kernels[i], cl::NullRange, cl::NDRange(1), cl::NDRange(1), &add_start_events, &add_events[i]);
            }

            cl::UserEvent triad_user_event(*config.context, &err);
            ASSERT_CL(err);
            std::vector<cl::Event> triad_start_events({triad_user_event});
            std::vector<cl::Event> triad_events(config.programSettings->kernelReplications);
            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                command_queues[i].enqueueNDRangeKernel(triad_kernels[i], cl::NullRange, cl::NDRange(1), cl::NDRange(1), &triad_start_events, &triad_events[i]);
            }

            startExecution = std::chrono::high_resolution_clock::now();
            copy_user_event.setStatus(CL_COMPLETE);
            cl::Event::waitForEvents(copy_events);

            endExecution = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>
                    (endExecution - startExecution);
            timingMap[COPY_KEY].push_back(duration.count());

            startExecution = std::chrono::high_resolution_clock::now();

            scale_user_event.setStatus(CL_COMPLETE);
            cl::Event::waitForEvents(scale_events);

            endExecution = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>
                    (endExecution - startExecution);
            timingMap[SCALE_KEY].push_back(duration.count());

            startExecution = std::chrono::high_resolution_clock::now();

            add_user_event.setStatus(CL_COMPLETE);
            cl::Event::waitForEvents(add_events);

            endExecution = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>
                    (endExecution - startExecution);
            timingMap[ADD_KEY].push_back(duration.count());

            startExecution = std::chrono::high_resolution_clock::now();

            triad_user_event.setStatus(CL_COMPLETE);
            cl::Event::waitForEvents(triad_events);

            endExecution = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>
                    (endExecution - startExecution);
            timingMap[TRIAD_KEY].push_back(duration.count());

            startExecution = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
#ifdef USE_SVM
                clEnqueueSVMUnmap(command_queues[i](),
                            reinterpret_cast<void *>(&A[data_per_kernel * i]), 0,
                            NULL, NULL);
                clEnqueueSVMUnmap(command_queues[i](),
                            reinterpret_cast<void *>(&B[data_per_kernel * i]), 0,
                            NULL, NULL);
                clEnqueueSVMUnmap(command_queues[i](),
                            reinterpret_cast<void *>(&C[data_per_kernel * i]), 0,
                            NULL, NULL);
#else
                command_queues[i].enqueueReadBuffer(Buffers_A[i], CL_FALSE, 0,
                                                    sizeof(HOST_DATA_TYPE) * data_per_kernel,
                                                    &A[data_per_kernel * i]);
                command_queues[i].enqueueReadBuffer(Buffers_B[i], CL_FALSE, 0,
                                                    sizeof(HOST_DATA_TYPE) * data_per_kernel,
                                                    &B[data_per_kernel * i]);
                command_queues[i].enqueueReadBuffer(Buffers_C[i], CL_FALSE, 0,
                                                    sizeof(HOST_DATA_TYPE) * data_per_kernel,
                                                    &C[data_per_kernel * i]);
#endif
            }

            for (int i = 0; i < config.programSettings->kernelReplications; i++) {
                command_queues[i].finish();
            }

            endExecution = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>
                    (endExecution - startExecution);
            timingMap[PCIE_READ_KEY].push_back(duration.count());

        }

        std::unique_ptr<stream::StreamExecutionTimings> result(new stream::StreamExecutionTimings{
                timingMap,
                config.programSettings->streamArraySize
        });
        return result;
    }

    bool initialize_queues_and_kernels(const hpcc_base::ExecutionSettings<stream::StreamProgramSettings> &config,
                                       unsigned int data_per_kernel, const std::vector<cl::Buffer> &Buffers_A,
                                       const std::vector<cl::Buffer> &Buffers_B,
                                       const std::vector<cl::Buffer> &Buffers_C,
                                       std::vector<cl::Kernel> &test_kernels, std::vector<cl::Kernel> &copy_kernels,
                                       std::vector<cl::Kernel> &scale_kernels, std::vector<cl::Kernel> &add_kernels,
                                       std::vector<cl::Kernel> &triad_kernels,
                                       std::vector<cl::CommandQueue> &command_queues) {
        int err;
        for (int i=0; i < config.programSettings->kernelReplications; i++) {
            // create the kernels
            cl::Kernel testkernel(*config.program, ("scale_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel copykernel(*config.program, ("copy_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel scalekernel(*config.program, ("scale_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel addkernel(*config.program, ("add_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel triadkernel(*config.program, ("triad_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);

            HOST_DATA_TYPE scalar = static_cast<HOST_DATA_TYPE>(3.0);
            HOST_DATA_TYPE test_scalar = static_cast<HOST_DATA_TYPE>(2.0);
            //prepare kernels
            err = testkernel.setArg(0, Buffers_A[i]);
            ASSERT_CL(err);
            err = testkernel.setArg(1, Buffers_A[i]);
            ASSERT_CL(err);
            err = testkernel.setArg(2, test_scalar);
            ASSERT_CL(err);
            err = testkernel.setArg(3, data_per_kernel);
            ASSERT_CL(err);
            //set arguments of copy kernel
            err = copykernel.setArg(0, Buffers_A[i]);
            ASSERT_CL(err);
            err = copykernel.setArg(1, Buffers_C[i]);
            ASSERT_CL(err);
            err = copykernel.setArg(2, data_per_kernel);
            ASSERT_CL(err);
            //set arguments of scale kernel
            err = scalekernel.setArg(0, Buffers_C[i]);
            ASSERT_CL(err);
            err = scalekernel.setArg(1, Buffers_B[i]);
            ASSERT_CL(err);
            err = scalekernel.setArg(2, scalar);
            ASSERT_CL(err);
            err = scalekernel.setArg(3, data_per_kernel);
            ASSERT_CL(err);
            //set arguments of add kernel
            err = addkernel.setArg(0, Buffers_A[i]);
            ASSERT_CL(err);
            err = addkernel.setArg(1, Buffers_B[i]);
            ASSERT_CL(err);
            err = addkernel.setArg(2, Buffers_C[i]);
            ASSERT_CL(err);
            err = addkernel.setArg(3, data_per_kernel);
            ASSERT_CL(err);
            //set arguments of triad kernel
            err = triadkernel.setArg(0, Buffers_B[i]);
            ASSERT_CL(err);
            err = triadkernel.setArg(1, Buffers_C[i]);
            ASSERT_CL(err);
            err = triadkernel.setArg(2, Buffers_A[i]);
            ASSERT_CL(err);
            err = triadkernel.setArg(3, scalar);
            ASSERT_CL(err);
            err = triadkernel.setArg(4, data_per_kernel);
            ASSERT_CL(err);

            command_queues.push_back(cl::CommandQueue(*config.context));
            test_kernels.push_back(testkernel);
            copy_kernels.push_back(copykernel);
            scale_kernels.push_back(scalekernel);
            add_kernels.push_back(addkernel);
            triad_kernels.push_back(triadkernel);
        }

        return true;
    }

    bool initialize_queues_and_kernels_single(const hpcc_base::ExecutionSettings<stream::StreamProgramSettings> &config,
                                       unsigned int data_per_kernel, const std::vector<cl::Buffer> &Buffers_A,
                                       const std::vector<cl::Buffer> &Buffers_B,
                                       const std::vector<cl::Buffer> &Buffers_C,
                                       std::vector<cl::Kernel> &test_kernels, std::vector<cl::Kernel> &copy_kernels,
                                       std::vector<cl::Kernel> &scale_kernels, std::vector<cl::Kernel> &add_kernels,
                                       std::vector<cl::Kernel> &triad_kernels,
                                       HOST_DATA_TYPE* A,
                                       HOST_DATA_TYPE* B,
                                       HOST_DATA_TYPE* C,
                                       std::vector<cl::CommandQueue> &command_queues) {
        int err;
        for (int i=0; i < config.programSettings->kernelReplications; i++) {
#ifdef INTEL_FPGA
            // create the kernels
            cl::Kernel testkernel(*config.program, ("calc_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel copykernel(*config.program, ("calc_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel scalekernel(*config.program, ("calc_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel addkernel(*config.program, ("calc_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel triadkernel(*config.program, ("calc_" + std::to_string(i)).c_str(), &err);
            ASSERT_CL(err);
#endif
#ifdef XILINX_FPGA
            // create the kernels
            cl::Kernel testkernel(*config.program, ("calc_0:{calc_0_" + std::to_string(i+1) + "}").c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel copykernel(*config.program, ("calc_0:{calc_0_" + std::to_string(i+1) + "}").c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel scalekernel(*config.program, ("calc_0:{calc_0_" + std::to_string(i+1) + "}").c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel addkernel(*config.program, ("calc_0:{calc_0_" + std::to_string(i+1) + "}").c_str(), &err);
            ASSERT_CL(err);
            cl::Kernel triadkernel(*config.program, ("calc_0:{calc_0_" + std::to_string(i+1) + "}").c_str(), &err);
            ASSERT_CL(err);
#endif
            HOST_DATA_TYPE scalar = static_cast<HOST_DATA_TYPE>(3.0);
            HOST_DATA_TYPE test_scalar = static_cast<HOST_DATA_TYPE>(2.0);
            //prepare kernels
#ifdef USE_SVM
            err = clSetKernelArgSVMPointer(testkernel(), 0,
                                        reinterpret_cast<void*>(A));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(testkernel(), 1,
                                        reinterpret_cast<void*>(A));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(testkernel(), 2,
                                        reinterpret_cast<void*>(A));
            ASSERT_CL(err);
#else
            err = testkernel.setArg(0, Buffers_A[i]);
            ASSERT_CL(err);
            err = testkernel.setArg(1, Buffers_A[i]);
            ASSERT_CL(err);
            err = testkernel.setArg(2, Buffers_A[i]);
            ASSERT_CL(err);
#endif
            err = testkernel.setArg(3, test_scalar);
            ASSERT_CL(err);
            err = testkernel.setArg(4, data_per_kernel);
            ASSERT_CL(err);
            err = testkernel.setArg(5, SCALE_KERNEL_TYPE);
            ASSERT_CL(err);

            //set arguments of copy kernel
#ifdef USE_SVM
            err = clSetKernelArgSVMPointer(copykernel(), 0,
                                        reinterpret_cast<void*>(A));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(copykernel(), 1,
                                        reinterpret_cast<void*>(A));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(copykernel(), 2,
                                        reinterpret_cast<void*>(C));
            ASSERT_CL(err);
#else
            err = copykernel.setArg(0, Buffers_A[i]);
            ASSERT_CL(err);
            err = copykernel.setArg(1, Buffers_A[i]);
            ASSERT_CL(err);
            err = copykernel.setArg(2, Buffers_C[i]);
            ASSERT_CL(err);
#endif
            err = copykernel.setArg(3, static_cast<HOST_DATA_TYPE>(1.0));
            ASSERT_CL(err);
            err = copykernel.setArg(4, data_per_kernel);
            ASSERT_CL(err);
            err = copykernel.setArg(5, COPY_KERNEL_TYPE);
            ASSERT_CL(err);
            //set arguments of scale kernel
#ifdef USE_SVM
            err = clSetKernelArgSVMPointer(scalekernel(), 0,
                                        reinterpret_cast<void*>(C));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(scalekernel(), 1,
                                        reinterpret_cast<void*>(C));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(scalekernel(), 2,
                                        reinterpret_cast<void*>(B));
            ASSERT_CL(err);
#else
            err = scalekernel.setArg(0, Buffers_C[i]);
            ASSERT_CL(err);
            err = scalekernel.setArg(1, Buffers_C[i]);
            ASSERT_CL(err);
            err = scalekernel.setArg(2, Buffers_B[i]);
            ASSERT_CL(err);
#endif
            err = scalekernel.setArg(3, scalar);
            ASSERT_CL(err);
            err = scalekernel.setArg(4, data_per_kernel);
            ASSERT_CL(err);
            err = scalekernel.setArg(5, SCALE_KERNEL_TYPE);
            ASSERT_CL(err);
            //set arguments of add kernel
#ifdef USE_SVM
            err = clSetKernelArgSVMPointer(addkernel(), 0,
                                        reinterpret_cast<void*>(A));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(addkernel(), 1,
                                        reinterpret_cast<void*>(B));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(addkernel(), 2,
                                        reinterpret_cast<void*>(C));
            ASSERT_CL(err);
#else
            err = addkernel.setArg(0, Buffers_A[i]);
            ASSERT_CL(err);
            err = addkernel.setArg(1, Buffers_B[i]);
            ASSERT_CL(err);
            err = addkernel.setArg(2, Buffers_C[i]);
            ASSERT_CL(err);
#endif
            err = addkernel.setArg(3, static_cast<HOST_DATA_TYPE>(1.0));
            ASSERT_CL(err);
            err = addkernel.setArg(4, data_per_kernel);
            ASSERT_CL(err);
            err = addkernel.setArg(5, ADD_KERNEL_TYPE);
            ASSERT_CL(err);
            //set arguments of triad kernel
#ifdef USE_SVM
            err = clSetKernelArgSVMPointer(triadkernel(), 0,
                                        reinterpret_cast<void*>(C));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(triadkernel(), 1,
                                        reinterpret_cast<void*>(B));
            ASSERT_CL(err);
            err = clSetKernelArgSVMPointer(triadkernel(), 2,
                                        reinterpret_cast<void*>(A));
            ASSERT_CL(err);
#else
            err = triadkernel.setArg(0, Buffers_C[i]);
            ASSERT_CL(err);
            err = triadkernel.setArg(1, Buffers_B[i]);
            ASSERT_CL(err);
            err = triadkernel.setArg(2, Buffers_A[i]);
            ASSERT_CL(err);
#endif
            err = triadkernel.setArg(3, scalar);
            ASSERT_CL(err);
            err = triadkernel.setArg(4, data_per_kernel);
            ASSERT_CL(err);
            err = triadkernel.setArg(5, TRIAD_KERNEL_TYPE);
            ASSERT_CL(err);

            command_queues.push_back(cl::CommandQueue(*config.context));
            test_kernels.push_back(testkernel);
            copy_kernels.push_back(copykernel);
            scale_kernels.push_back(scalekernel);
            add_kernels.push_back(addkernel);
            triad_kernels.push_back(triadkernel);
        }

        return true;
    }

    void initialize_buffers(const hpcc_base::ExecutionSettings<stream::StreamProgramSettings> &config, unsigned int data_per_kernel,
                            std::vector<cl::Buffer> &Buffers_A, std::vector<cl::Buffer> &Buffers_B,
                            std::vector<cl::Buffer> &Buffers_C) {
        unsigned mem_bits = CL_MEM_READ_WRITE;
#ifdef INTEL_FPGA
#ifdef USE_HBM
        mem_bits |= CL_MEM_HETEROGENEOUS_INTELFPGA;
#endif
#endif

        if (!config.programSettings->useMemoryInterleaving) {
            //Create Buffers for input and output
            for (int i=0; i < config.programSettings->kernelReplications; i++) {
#if defined(INTEL_FPGA) && !defined(USE_HBM)
                if (config.programSettings->useSingleKernel) {
                    //Create Buffers for input and output
                    Buffers_A.push_back(cl::Buffer(*config.context, mem_bits | ((i + 1) << 16), sizeof(HOST_DATA_TYPE)*data_per_kernel));
                    Buffers_B.push_back(cl::Buffer(*config.context, mem_bits | ((i + 1) << 16), sizeof(HOST_DATA_TYPE)*data_per_kernel));
                    Buffers_C.push_back(cl::Buffer(*config.context, mem_bits | ((i + 1) << 16), sizeof(HOST_DATA_TYPE)*data_per_kernel));
                }
                else {
                    //Create Buffers for input and output
                    Buffers_A.push_back(cl::Buffer(*config.context, mem_bits | CL_CHANNEL_1_INTELFPGA, sizeof(HOST_DATA_TYPE)*data_per_kernel));
                    Buffers_B.push_back(cl::Buffer(*config.context, mem_bits | CL_CHANNEL_3_INTELFPGA, sizeof(HOST_DATA_TYPE)*data_per_kernel));
                    Buffers_C.push_back(cl::Buffer(*config.context, mem_bits | CL_CHANNEL_2_INTELFPGA, sizeof(HOST_DATA_TYPE)*data_per_kernel));
                }
#endif
#if defined(XILINX_FPGA) || defined(USE_HBM)
                Buffers_A.push_back(cl::Buffer(*config.context, mem_bits, sizeof(HOST_DATA_TYPE)*data_per_kernel));
                Buffers_B.push_back(cl::Buffer(*config.context, mem_bits, sizeof(HOST_DATA_TYPE)*data_per_kernel));
                Buffers_C.push_back(cl::Buffer(*config.context, mem_bits, sizeof(HOST_DATA_TYPE)*data_per_kernel));
#endif
            }

        } else {
            for (int i=0; i < config.programSettings->kernelReplications; i++) {
                //Create Buffers for input and output
                Buffers_A.push_back(cl::Buffer(*config.context, mem_bits, sizeof(HOST_DATA_TYPE)*data_per_kernel));
                Buffers_B.push_back(cl::Buffer(*config.context, mem_bits, sizeof(HOST_DATA_TYPE)*data_per_kernel));
                Buffers_C.push_back(cl::Buffer(*config.context, mem_bits, sizeof(HOST_DATA_TYPE)*data_per_kernel));
            }
        }
    }

}  // namespace bm_execution
