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

#ifdef INTEL_FPGA
#pragma OPENCL EXTENSION cl_intel_channels : enable
#endif

#define min(a,b) (a<b?a:b)

#define LOGN            LOG_FFT_SIZE

#define LOGPOINTS       3
#define POINTS          (1 << LOGPOINTS)

// Log of how much to fetch at once for one area of input buffer.
// LOG_CONT_FACTOR_LIMIT computation makes sure that C_LEN below
// is non-negative. Keep it bounded by 6, as going larger will waste
// on-chip resources but won't give performance gains.
#define LOG_CONT_FACTOR_LIMIT1 (LOGN - (2 * (LOGPOINTS)))
#define LOG_CONT_FACTOR_LIMIT2 (((LOG_CONT_FACTOR_LIMIT1) >= 0) ? (LOG_CONT_FACTOR_LIMIT1) : 0)
#define LOG_CONT_FACTOR        (((LOG_CONT_FACTOR_LIMIT2) <= 6) ? (LOG_CONT_FACTOR_LIMIT1) : 6)
#define CONT_FACTOR            (1 << LOG_CONT_FACTOR)

// Need some depth to our channels to accommodate their bursty filling.
#ifdef INTEL_FPGA
channel float2 chanin[POINTS] __attribute__((depth(CONT_FACTOR*POINTS)));
#endif
#ifdef XILINX_FPGA
pipe float2x8 chanin __attribute__((xcl_reqd_pipe_depth(CONT_FACTOR*POINTS)));
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

// fetch N points as follows:
// - each thread will load 8 consecutive values
// - load CONT_FACTOR consecutive loads (8 values each), then jump by N/8, and load next
//   CONT_FACTOR consecutive values.
// - Once load CONT_FACTOR values starting at 7N/8, send CONT_FACTOR values
//   into the channel to the fft kernel.
// - start process again. 
// This way, only need 8xCONT_FACTOR local memory buffer, instead of 8xN.
//
// Group index is used as follows ( 0 to CONT_FACTOR, iteration num )
//
//  64K values = 2^16, num_fetches=2^13, 2^6 = CONT_FACTOR, 2^7=num_Fetches / cont_factor
//
// <   C ><B><  A >
// 5432109876543210
//  A -- fetch within contiguous block
//  B -- B * N/8 region selector
//  C -- num times fetch cont_factor * 8 values (or num times fill the buffer)

// INPUT GID POINTS
// C_LEN must be at least 0. Can't be negative.
#define A_START 0
#define A_END   (LOG_CONT_FACTOR + LOGPOINTS - 1)

#define B_START (A_END + 1)
#define B_END   (B_START + LOGPOINTS - 1)

#define C_START (B_END + 1)
#define C_END   (LOGN - 1)

#define D_START (C_END + 1)
#define D_END   31

#define A_LEN   (A_END - A_START + 1)
#define B_LEN   (B_END - B_START + 1)
#define C_LEN   (C_END - C_START + 1)
#define D_LEN   (D_END - D_START + 1)
#define EXTRACT(id,start,len) ((id >> start) & ((1 << len) - 1))

uint permute_gid (uint gid) {
  uint result = 0;
  // result[31:16]= gid[31:16] = D
  // result[15:13] = gid[10:8] = C
  // result[12:8]  = gid[15:11] = B
  // result[7:0]  = gid[10:0] = A

  uint A = EXTRACT(gid, A_START, A_LEN);
  uint B = EXTRACT(gid, B_START, B_LEN);
  uint C = EXTRACT(gid, C_START, C_LEN);
  uint D = EXTRACT(gid, D_START, D_LEN);
    
  // swap B and C
  uint new_c_start = A_END + 1;
  uint new_b_start = new_c_start + C_LEN;
  result = (D << D_START) | (B << new_b_start) | (C << new_c_start) | (A << A_START);
  return result;
}

#ifdef INTEL_FPGA
// group dimension (N/(8*CONT_FACTOR), num_iterations)
__kernel
__attribute__((reqd_work_group_size(CONT_FACTOR * POINTS, 1, 1)))
void fetch (global float2 * restrict src) {

  const int N = (1 << LOGN);
  // Each thread will fetch POINTS points. Need POINTS times to pass to FFT.
  const int BUF_SIZE = 1 << (LOG_CONT_FACTOR + LOGPOINTS + LOGPOINTS);

  // Local memory for CONT_FACTOR * POINTS points
  local float2 buf[BUF_SIZE];

  uint iteration = get_global_id(1);
  uint group_per_iter = get_global_id(0);
  
  // permute global addr but not the local addr
  uint global_addr = iteration * N + group_per_iter;
  global_addr = permute_gid (global_addr << LOGPOINTS);
  uint lid = get_local_id(0);
  uint local_addr = lid << LOGPOINTS;

__attribute__((opencl_unroll_hint(POINTS)))
  for (uint k = 0; k < POINTS; k++) {
    buf[local_addr + k] = src[global_addr + k];
  }

  barrier (CLK_LOCAL_MEM_FENCE);

#ifdef XILINX_FPGA
  float2 buf2[POINTS];
  float2x8 buf2x8;
#endif

__attribute__((opencl_unroll_hint(POINTS)))
  for (uint k = 0; k < POINTS; k++) {
    uint buf_addr = bit_reversed(k,LOGPOINTS) * CONT_FACTOR * POINTS + lid;
    #ifdef INTEL_FPGA
    write_channel_intel (chanin[k], buf[buf_addr]);
    #else
    buf2[k] = buf[buf_addr];
    #endif
  }
  #ifdef XILINX_FPGA
  buf2x8.i0 = buf2[0];
  buf2x8.i1 = buf2[1];
  buf2x8.i2 = buf2[2];
  buf2x8.i3 = buf2[3];
  buf2x8.i4 = buf2[4];
  buf2x8.i5 = buf2[5];
  buf2x8.i6 = buf2[6];
  buf2x8.i7 = buf2[7];
  write_pipe_block(chanin, &buf2x8);
  #endif
}
#endif

#define BUFFER_REPLICATION 4

#ifdef XILINX_FPGA
__kernel
__attribute__ ((max_global_work_dim(0), reqd_work_group_size(1,1,1)))
void fetch(__global float2 * restrict src, int iter) {

  const int N = (1 << LOGN);

  // Duplicated input buffer. One will be used to write and buffer data from global memory
  // The other will be used to read and forward the data over the channels.
  // Read and write buffers will be swapped in every iteration
  float2 buf[BUFFER_REPLICATION][POINTS][N / POINTS] __attribute__((xcl_array_partition(cyclic, 2, 1), xcl_array_partition(complete, 2), xcl_array_partition(cyclic, POINTS, 3)));

  // for iter iterations and one additional iteration to empty the last buffer
  __attribute__((xcl_loop_tripcount(2*(N / POINTS),5000*(N / POINTS),100*(N / POINTS))))
  for(unsigned k = 0; k < (iter + 1) * (N / POINTS); k++){ 

    // Read the next 8 values from global memory
    // in the last iteration just read garbage, but the data will not be forwarded over the pipes.
    // This allows the use of memory bursts here.
    __attribute__((opencl_unroll_hint(POINTS)))
    for(int j = 0; j < POINTS; j++){
      unsigned local_i = ((k << LOGPOINTS) + j) & (N - 1);
      buf[(k >> (LOGN - LOGPOINTS)) & (BUFFER_REPLICATION - 1)][local_i >> (LOGN - LOGPOINTS)][local_i & ((1 << (LOGN - LOGPOINTS)) - 1)] = src[(k << LOGPOINTS) + j];
    }

    // Start in the second iteration to forward the buffered data over the pipe
    if (k >= (N / POINTS)) {
#ifdef INTEL_FPGA
      __attribute__((opencl_unroll_hint(POINTS)))
      for (uint j = 0; j < POINTS; j++) {
        write_channel_intel (chanin[j], buf[((k >> LOGN) - 1) & (BUFFER_REPLICATION - 1)][bit_reversed(j,LOGPOINTS)][k & ((1 << (LOGN - LOGPOINTS)) - 1)]);
      }
#endif
#ifdef XILINX_FPGA
      float2x8 buf2x8;
      buf2x8.i0 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][0][k & ((1 << (LOGN - LOGPOINTS)) - 1)];          
      buf2x8.i1 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][4][k & ((1 << (LOGN - LOGPOINTS)) - 1)];  
      buf2x8.i2 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][2][k & ((1 << (LOGN - LOGPOINTS)) - 1)];  
      buf2x8.i3 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][6][k & ((1 << (LOGN - LOGPOINTS)) - 1)]; 
      buf2x8.i4 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][1][k & ((1 << (LOGN - LOGPOINTS)) - 1)]; 
      buf2x8.i5 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][5][k & ((1 << (LOGN - LOGPOINTS)) - 1)];
      buf2x8.i6 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][3][k & ((1 << (LOGN - LOGPOINTS)) - 1)];
      buf2x8.i7 = buf[((k >> (LOGN - LOGPOINTS)) - 1) & (BUFFER_REPLICATION - 1)][7][k & ((1 << (LOGN - LOGPOINTS)) - 1)];

      write_pipe_block(chanin, &buf2x8);
#endif
    }
  }
}
#endif



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
kernel void fft1d(global float2 * restrict dest,
                  int count, int inverse) {

  const int N = (1 << LOGN);

  /* The FFT engine requires a sliding window array for data reordering; data 
   * stored in this array is carried across loop iterations and shifted by one 
   * element every iteration; all loop dependencies derived from the uses of 
   * this array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  /* This is the main loop. It runs 'count' back-to-back FFT transforms
   * In addition to the 'count * (N / 8)' iterations, it runs 'N / 8 - 1'
   * additional iterations to drain the last outputs 
   * (see comments attached to the FFT engine)
   *
   * The compiler leverages pipeline parallelism by overlapping the 
   * iterations of this loop - launching one iteration every clock cycle
   */
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
      data.i0 = read_channel_intel(chanin[0]);
      data.i1 = read_channel_intel(chanin[1]);
      data.i2 = read_channel_intel(chanin[2]);
      data.i3 = read_channel_intel(chanin[3]);
      data.i4 = read_channel_intel(chanin[4]);
      data.i5 = read_channel_intel(chanin[5]);
      data.i6 = read_channel_intel(chanin[6]);
      data.i7 = read_channel_intel(chanin[7]);
      #else
      read_pipe_block(chanin, &data);
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
      int base = POINTS * (i - (N / POINTS - 1));
 
      // These consecutive accesses will be coalesced by the compiler
      dest[base] = data.i0;
      dest[base + 1] = data.i1;
      dest[base + 2] = data.i2;
      dest[base + 3] = data.i3;
      dest[base + 4] = data.i4;
      dest[base + 5] = data.i5;
      dest[base + 6] = data.i6;
      dest[base + 7] = data.i7;
    }
  }
}

