// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

/* This is the top-level device source file for the fft1d example. The code is
 * written as an OpenCL single work-item kernel. This coding style allows the 
 * compiler to extract loop-level parallelism from the source code and 
 * instantiate a hardware pipeline capable of executing concurrently a large 
 * number of loop iterations. The compiler analyses loop-carried dependencies, 
 * and these translate into data transfers across concurrently executed loop 
 * iterations. 
 *
 * Careful coding ensures that all loop-carried dependencies are trivial, 
 * merely data transfers which span a single clock cycle. The FFT algorithm 
 * requires passing data forward across loop iterations. The code uses a 
 * sliding window to implement these data transfers. The advantage of using a
 * sliding window is that dependencies across consecutive loop iterations have
 * an invariant source and destination (pairs of constant offset array 
 * elements). Such transfer patterns can be implemented efficiently by the 
 * FPGA hardware. All this ensures an overall processing a throughput of one 
 * loop iteration per clock cycle.
 *
 * The size of the FFT transform can be customized via an argument to the FFT 
 * engine. This argument has to be a compile time constant to ensure that the 
 * compiler can propagate it throughout the function body and generate 
 * efficient hardware.
 */

// Include source code for an engine that produces 8 points each step
#include "fft_8.cl"

#include "parameters.h"


#define min(a,b) (a<b?a:b)

#define LOGN            LOG_FFT_SIZE

#define LOGPOINTS       3
#define POINTS          (1 << LOGPOINTS)

// Need some depth to our channels to accommodate their bursty filling.
#ifdef INTEL_FPGA
#pragma OPENCL EXTENSION cl_intel_channels : enable
// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_total_replications)]
channel float2 chanin/*PY_CODE_GEN i*/[POINTS] __attribute__((depth(POINTS)));
// PY_CODE_GEN block_end
#endif
#ifdef XILINX_FPGA
#define XILINX_PIPE_DEPTH 16
//#define XILINX_PIPE_DEPTH ((1 << (LOGN - LOGPOINTS) < 16) ? 16 : (1 << (LOGN - LOGPOINTS)))

// Compiler states, that the pipe depth needs at least to be 16
// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_total_replications)]
pipe float2x8 chanin/*PY_CODE_GEN i*/ __attribute__((xcl_reqd_pipe_depth(XILINX_PIPE_DEPTH)));
pipe float2x8 chanout/*PY_CODE_GEN i*/ __attribute__((xcl_reqd_pipe_depth(XILINX_PIPE_DEPTH)));
// PY_CODE_GEN block_end
#endif

uint bit_reversed(uint x, uint bits) {
  uint y = 0;
__attribute__((opencl_unroll_hint()))
  for (uint i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  y &= ((1 << bits) - 1);
  return y;
}

// PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(num_total_replications)]

__kernel
__attribute__ ((max_global_work_dim(0), reqd_work_group_size(1,1,1)))
void fetch/*PY_CODE_GEN i*/(__global float2 * restrict src, int iter) {

  const int N = (1 << LOGN);

  // Input buffer that can hold the data for two FFTs
  float2 buf[2*N/POINTS][POINTS] __attribute__((numbanks(POINTS),xcl_array_partition(block, N/POINTS, 1), xcl_array_partition(complete, 2)));

  // for iter iterations and one additional iteration to empty the last buffer
  __attribute__((xcl_loop_tripcount(2*(N / POINTS),5000*(N / POINTS),100*(N / POINTS))))
  for(unsigned k = 0; k < (iter + 1) * (N / POINTS); k++){ 

    float2 read_chunk[POINTS];

    // Read the next 8 values from global memory
    // in the last iteration just read garbage, but the data will not be forwarded over the pipes.
    // This allows the use of memory bursts here.
    // Also the data is shifted  every N/POINTS/POINTS iterations
    __attribute__((opencl_unroll_hint(POINTS)))
    for(int j = 0; j < POINTS; j++){
      unsigned shifts = (k << LOGPOINTS) >> (LOGN - LOGPOINTS);
      unsigned final_buffer_pos = (j + shifts) & (POINTS - 1);
      read_chunk[final_buffer_pos] = src[(k << LOGPOINTS) + j];
    }

    // Write the shifted data into the memory buffer
    __attribute__((opencl_unroll_hint(POINTS)))
    for(int j = 0; j < POINTS; j++){
      unsigned local_i = ((k)) & (2* N/POINTS - 1);
      buf[local_i][j] = read_chunk[j];
    }

    if (k >= (N / POINTS)) {
      float2x8 buf2x8;

      unsigned offset = (((k - (N / POINTS)) >> (LOGN - LOGPOINTS)) << (LOGN - LOGPOINTS)) & (2*N/POINTS - 1);
      
      float2 write_chunk[POINTS];
      // Write the shifted data into the memory buffer
      __attribute__((opencl_unroll_hint(POINTS)))
      for(int j = 0; j < POINTS; j++){
        write_chunk[bit_reversed(j, LOGPOINTS)] = buf[offset + (j * (N/POINTS/POINTS) + ((k >> LOGPOINTS) & (N/POINTS/POINTS - 1)))][((k + j)     & (POINTS - 1))];
      }
#ifdef XILINX_FPGA
      buf2x8.i0 = write_chunk[0];          
      buf2x8.i1 = write_chunk[1];  
      buf2x8.i2 = write_chunk[2];  
      buf2x8.i3 = write_chunk[3]; 
      buf2x8.i4 = write_chunk[4]; 
      buf2x8.i5 = write_chunk[5];
      buf2x8.i6 = write_chunk[6];
      buf2x8.i7 = write_chunk[7];

      // Start in the second iteration to forward the buffered data over the pipe
      write_pipe_block(chanin/*PY_CODE_GEN i*/, &buf2x8);
#endif
#ifdef INTEL_FPGA
        write_channel_intel(chanin/*PY_CODE_GEN i*/[0], write_chunk[0]); 
        write_channel_intel(chanin/*PY_CODE_GEN i*/[1], write_chunk[1]);  
        write_channel_intel(chanin/*PY_CODE_GEN i*/[2], write_chunk[2]);  
        write_channel_intel(chanin/*PY_CODE_GEN i*/[3], write_chunk[3]);  
        write_channel_intel(chanin/*PY_CODE_GEN i*/[4], write_chunk[4]);  
        write_channel_intel(chanin/*PY_CODE_GEN i*/[5], write_chunk[5]); 
        write_channel_intel(chanin/*PY_CODE_GEN i*/[6], write_chunk[6]);  
        write_channel_intel(chanin/*PY_CODE_GEN i*/[7], write_chunk[7]);  
#endif
    }
  }
}



/* Attaching the attribute 'task' to the top level kernel to indicate 
 * that the host enqueues a task (a single work-item kernel)
 *
 * 'src' and 'dest' point to the input and output buffers in global memory; 
 * using restrict pointers as there are no dependencies between the buffers
 * 'count' represents the number of 4k sets to process
 * 'inverse' toggles between the direct and the inverse transform
 */

__attribute__ ((max_global_work_dim(0)))
__attribute__((reqd_work_group_size(1,1,1)))
kernel void fft1d/*PY_CODE_GEN i*/(
#ifdef INTEL_FPGA
                // Intel does not need a store kernel and directly writes back the result to global memory
                __global float2 * restrict dest,
#endif
                int count, int inverse) {

  const int N = (1 << LOGN);

  /* The FFT engine requires a sliding window array for data reordering; data 
   * stored in this array is carried across loop iterations and shifted by one 
   * element every iteration; all loop dependencies derived from the uses of 
   * this array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)] __attribute__((xcl_array_partition(complete, 0)));

  /* This is the main loop. It runs 'count' back-to-back FFT transforms
   * In addition to the 'count * (N / 8)' iterations, it runs 'N / 8 - 1'
   * additional iterations to drain the last outputs 
   * (see comments attached to the FFT engine)
   *
   * The compiler leverages pipeline parallelism by overlapping the 
   * iterations of this loop - launching one iteration every clock cycle
   */
   __attribute__((xcl_pipeline_loop(1)))
  for (unsigned i = 0; i < count * (N / POINTS) + N / POINTS - 1; i++) {

    /* As required by the FFT engine, gather input data from 8 distinct 
     * segments of the input buffer; for simplicity, this implementation 
     * does not attempt to coalesce memory accesses and this leads to 
     * higher resource utilization (see the fft2d example for advanced 
     * memory access techniques)
     */

    int base = (i / (N / POINTS)) * N;
    int offset = i % (N / POINTS);

    float2x8 data;
    // Perform memory transfers only when reading data in range
    if (i < count * (N / POINTS)) {
#ifdef INTEL_FPGA
      data.i0 = read_channel_intel(chanin/*PY_CODE_GEN i*/[0]);
      data.i1 = read_channel_intel(chanin/*PY_CODE_GEN i*/[1]);
      data.i2 = read_channel_intel(chanin/*PY_CODE_GEN i*/[2]);
      data.i3 = read_channel_intel(chanin/*PY_CODE_GEN i*/[3]);
      data.i4 = read_channel_intel(chanin/*PY_CODE_GEN i*/[4]);
      data.i5 = read_channel_intel(chanin/*PY_CODE_GEN i*/[5]);
      data.i6 = read_channel_intel(chanin/*PY_CODE_GEN i*/[6]);
      data.i7 = read_channel_intel(chanin/*PY_CODE_GEN i*/[7]);
#endif
#ifdef XILINX_FPGA
      read_pipe_block(chanin/*PY_CODE_GEN i*/, &data);
#endif
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Perform one step of the FFT engine
    data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

    /* Store data back to memory. FFT engine outputs are delayed by 
     * N / 8 - 1 steps, hence gate writes accordingly
     */
    if (i >= N / POINTS - 1) {
#ifdef INTEL_FPGA
      int base = POINTS * (i - (N / POINTS - 1));
 
      // These consecutive accesses will be coalesced by the compiler
      dest[base]     = data.i0;
      dest[base + 1] = data.i1;
      dest[base + 2] = data.i2;
      dest[base + 3] = data.i3;
      dest[base + 4] = data.i4;
      dest[base + 5] = data.i5;
      dest[base + 6] = data.i6;
      dest[base + 7] = data.i7;
#endif
#ifdef XILINX_FPGA
    // For Xilinx send the data to the store kernel to enable memory bursts
      write_pipe_block(chanout/*PY_CODE_GEN i*/, &data);
#endif
    }
  }
}

#ifdef XILINX_FPGA
/**
The store kernel just reads from the output channel and writes the data to the global memory.
This kernel works without conditional branches which enables memory bursts.
 */
__kernel
__attribute__ ((max_global_work_dim(0), reqd_work_group_size(1,1,1)))
void store/*PY_CODE_GEN i*/(__global float2 * restrict dest, int iter) {

  const int N = (1 << LOGN);

  // write the data back to global memory using memory bursts
  for(unsigned k = 0; k < iter * (N / POINTS); k++){ 
      float2x8 buf2x8;
      read_pipe_block(chanout/*PY_CODE_GEN i*/, &buf2x8);

      dest[(k << LOGPOINTS)]     = buf2x8.i0;    
      dest[(k << LOGPOINTS) + 1] = buf2x8.i1; 
      dest[(k << LOGPOINTS) + 2] = buf2x8.i2; 
      dest[(k << LOGPOINTS) + 3] = buf2x8.i3; 
      dest[(k << LOGPOINTS) + 4] = buf2x8.i4; 
      dest[(k << LOGPOINTS) + 5] = buf2x8.i5; 
      dest[(k << LOGPOINTS) + 6] = buf2x8.i6; 
      dest[(k << LOGPOINTS) + 7] = buf2x8.i7;    
  }
}
#endif

//PY_CODE_GEN block_end
