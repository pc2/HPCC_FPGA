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

// Complex single-precision floating-point radix-4 feedforward FFT / iFFT engine
// 
// See Mario Garrido, Jesús Grajal, M. A. Sanchez, Oscar Gustafsson:
// Pipeline Radix-2k Feedforward FFT Architectures. 
// IEEE Trans. VLSI Syst. 21(1): 23-32 (2013))
//
// The log(size) of the transform must be a compile-time constant argument. 
// This FFT engine processes 8 points for each invocation. The inputs are eight 
// ordered streams while the outputs are in bit reversed order.
//
// The entry point of the engine is the 'fft_step' function. This function
// passes 8 data points through a fixed sequence of processing blocks
// (butterfly, rotation, swap, reorder, multiplications, etc.) and produces
// 8 output points towards the overall FFT transform. 
//
// The engine is designed to be invoked from a loop in a single work-item task. 
// When compiling a single work-item task, the compiler leverages pipeline 
// parallelism and overlaps the execution of multiple invocations of this 
// function. A new instance can start processing every clock cycle


// Includes tabled twiddle factors - storing constants uses fewer resources
// than instantiating 'cos' or 'sin' hardware
#include "twid_radix4_8.cl" 

#include "parameters.h"

// Convenience struct representing the 8 data points processed each step
// Each member is a float2 representing a complex number
typedef struct {
   float2 i0;
   float2 i1;
   float2 i2;
   float2 i3;
   float2 i4;
   float2 i5;
   float2 i6;
   float2 i7;
} float2x8;

// FFT butterfly building block
__attribute__((always_inline))
float2x8 butterfly(float2x8 data) {
   float2x8 res;
   res.i0 = data.i0 + data.i1;
   res.i1 = data.i0 - data.i1;
   res.i2 = data.i2 + data.i3;
   res.i3 = data.i2 - data.i3;
   res.i4 = data.i4 + data.i5;
   res.i5 = data.i4 - data.i5;
   res.i6 = data.i6 + data.i7;
   res.i7 = data.i6 - data.i7;
   return res;
}

// Swap real and imaginary components in preparation for inverse transform
__attribute__((always_inline))
float2x8 swap_complex(float2x8 data) {
   float2x8 res;
   res.i0.x = data.i0.y;
   res.i0.y = data.i0.x;
   res.i1.x = data.i1.y;
   res.i1.y = data.i1.x;
   res.i2.x = data.i2.y;
   res.i2.y = data.i2.x;
   res.i3.x = data.i3.y;
   res.i3.y = data.i3.x;
   res.i4.x = data.i4.y;
   res.i4.y = data.i4.x;
   res.i5.x = data.i5.y;
   res.i5.y = data.i5.x;
   res.i6.x = data.i6.y;
   res.i6.y = data.i6.x;
   res.i7.x = data.i7.y;
   res.i7.y = data.i7.x;
   return res;
}

// FFT trivial rotation building block
__attribute__((always_inline))
float2x8 trivial_rotate(float2x8 data) {
   float2 tmp = data.i3;
   data.i3.x = tmp.y;
   data.i3.y = -tmp.x;
   tmp = data.i7;
   data.i7.x = tmp.y;
   data.i7.y = -tmp.x;
   return data;
}

// FFT data swap building block associated with trivial rotations
__attribute__((always_inline))
float2x8 trivial_swap(float2x8 data) {
   float2 tmp = data.i1;
   data.i1 = data.i2;
   data.i2 = tmp;
   tmp = data.i5;
   data.i5 = data.i6;
   data.i6 = tmp;
   return data;
}

// FFT data swap building block associated with complex rotations
__attribute__((always_inline))
float2x8 swap(float2x8 data) {
   float2 tmp = data.i1;
   data.i1 = data.i4;
   float2 tmp2 = data.i2;
   data.i2 = tmp;
   tmp = data.i3;
   data.i3 = data.i5;
   data.i4 = tmp2;
   data.i5 = data.i6;
   data.i6 = tmp;
   return data;
}

// This function "delays" the input by 'depth' steps
// Input 'data' from invocation N would be returned in invocation N + depth
// The 'shift_reg' sliding window is shifted by 1 element at every invocation 
__attribute__((always_inline))
float2 delay(float2 data, const int depth, float2 shift_reg[(1 << LOG_FFT_SIZE) + 8 * (LOG_FFT_SIZE - 2)], unsigned offset) {
   shift_reg[offset + depth] = data;
   return shift_reg[offset];
}

// FFT data reordering building block. Implements the reordering depicted below 
// (for depth = 2). The first valid outputs are in invocation 4
// Invocation count: 0123...          01234567...
// data.i0         : GECA...   ---->      DBCA...
// data.i1         : HFDB...   ---->      HFGE...
__attribute__((always_inline))
float2x8 reorder_data(float2x8 data, const int depth, float2 shift_reg[(1 << LOG_FFT_SIZE) + 8 * (LOG_FFT_SIZE - 2)], unsigned head_index, bool toggle) {
   // Use disconnected segments of length 'depth + 1' elements starting at 
   // 'shift_reg' to implement the delay elements. At the end of each FFT step, 
   // the contents of the entire buffer is shifted by 1 element
   data.i1 = delay(data.i1, depth, shift_reg, head_index);
   data.i3 = delay(data.i3, depth, shift_reg, head_index + depth + 1);
   data.i5 = delay(data.i5, depth, shift_reg, head_index + 2 * (depth + 1));
   data.i7 = delay(data.i7, depth, shift_reg, head_index + 3 * (depth + 1));
 
   if (toggle) {
      float2 tmp = data.i0;
      data.i0 = data.i1;
      data.i1 = tmp;
      tmp = data.i2;
      data.i2 = data.i3;
      data.i3 = tmp;
      tmp = data.i4;
      data.i4 = data.i5;
      data.i5 = tmp;
      tmp = data.i6;
      data.i6 = data.i7;
      data.i7 = tmp;
   }

   data.i0 = delay(data.i0, depth, shift_reg, head_index + 4 * (depth + 1));
   data.i2 = delay(data.i2, depth, shift_reg, head_index + 5 * (depth + 1));
   data.i4 = delay(data.i4, depth, shift_reg, head_index + 6 * (depth + 1));
   data.i6 = delay(data.i6, depth, shift_reg, head_index + 7 * (depth + 1));

   return data;
}

// Implements a complex number multiplication
__attribute__((always_inline))
float2 comp_mult(float2 a, float2 b) {
   float2 res;
   res.x = a.x * b.x - a.y * b.y;
   res.y = a.x * b.y + a.y * b.x;
   return res;
}

// Produces the twiddle factor associated with a processing stream 'stream', 
// at a specified 'stage' during a step 'index' of the computation
//
// If there are precomputed twiddle factors for the given FFT size, uses them
// This saves hardware resources, because it avoids evaluating 'cos' and 'sin'
// functions
__attribute__((always_inline))
float2 twiddle(int index, int stage, int size, int stream) {
   float2 twid;
   // Twiddle facotr array taken from twid_radix4_8.cl!
   // Use the precomputed twiddle factors, if available - otherwise, compute them
   int twid_stage = stage >> 1;
   if (size <= (1 << (TWID_STAGES * 2 + 2))) {
      twid.x = twiddles_cos[twid_stage][stream]
                                  [index * ((1 << (TWID_STAGES * 2 + 2)) / size)];
      twid.y = twiddles_sin[twid_stage][stream]
                                  [index * ((1 << (TWID_STAGES * 2 + 2)) / size)];
   } else {
      // This would generate hardware consuming a large number of resources
      // Instantiated only if precomputed twiddle factors are available
      const float TWOPI = 2.0f * M_PI_F;
      int multiplier;

      // The latter 3 streams will generate the second half of the elements
      // In that case phase = 1
      
      int phase = 0;
      if (stream >= 3) {
         stream -= 3; 
         phase = 1;
      }
      switch (stream) {
         case 0: multiplier = 2; break;
         case 1: multiplier = 1; break;
         case 2: multiplier = 3; break;
         default: multiplier = 0;
      }
      int pos = (1 << (stage - 1)) * multiplier * ((index + (size / 8) * phase) 
                                          & (size / 4 / (1 << (stage - 1)) - 1));
      float theta = -1.0f * TWOPI / size * (pos & (size - 1));
      twid.x = native_cos(theta);
      twid.y = native_sin(theta);
   }
   return twid;
}

// FFT complex rotation building block
__attribute__((always_inline))
float2x8 complex_rotate(float2x8 data, int index, int stage, int size) {
   data.i1 = comp_mult(data.i1, twiddle(index, stage, size, 0));
   data.i2 = comp_mult(data.i2, twiddle(index, stage, size, 1));
   data.i3 = comp_mult(data.i3, twiddle(index, stage, size, 2));
   data.i5 = comp_mult(data.i5, twiddle(index, stage, size, 3));
   data.i6 = comp_mult(data.i6, twiddle(index, stage, size, 4));
   data.i7 = comp_mult(data.i7, twiddle(index, stage, size, 5));
   return data;
}


// Process 8 input points towards and a FFT/iFFT of size N, N >= 8 
// (in order input, bit reversed output). Apply all input points in N / 8 
// consecutive invocations. Obtain all outputs in N /8 consecutive invocations 
// starting with invocation N /8 - 1 (outputs are delayed). Multiple back-to-back 
// transforms can be executed
//
// 'data' encapsulates 8 complex single-precision floating-point input points
// 'step' specifies the index of the current invocation 
// 'fft_delay_elements' is an array representing a sliding window of size N+8*(log(N)-2)
// 'inverse' toggles between the direct and inverse transform
// 'logN' should be a COMPILE TIME constant evaluating log(N) - the constant is 
//        propagated throughout the code to achieve efficient hardware
//
__attribute__((always_inline))
float2x8 fft_step(float2x8 data, int step, float2 fft_delay_elements[(1 << LOG_FFT_SIZE) + 8 * (LOG_FFT_SIZE - 2)], 
                  bool inverse, const int logN) {

    const int size = 1 << logN;
    // Swap real and imaginary components if doing an inverse transform
    if (inverse) {
       data = swap_complex(data);
    }

    // Stage 0 of feed-forward FFT
    data = butterfly(data);
    data = trivial_rotate(data);
    data = trivial_swap(data);
    
    // Stage 1
    data = butterfly(data);
    data = complex_rotate(data, step & (size / 8 - 1), 1, size);
    data = swap(data);

    // Next logN - 2 stages alternate two computation patterns - represented as
    // a loop to avoid code duplication. Instruct the compiler to fully unroll 
    // the loop to increase the  amount of pipeline parallelism and allow feed 
    // forward execution

   __attribute__((opencl_unroll_hint(LOG_FFT_SIZE - 3)))
    for (int stage = 2; stage < logN - 1; stage++) {
        bool complex_stage = stage & 1; // stages 3, 5, ...

        // Figure out the index of the element processed at this stage
        // Subtract (add modulo size / 8) the delay incurred as data travels 
        // from one stage to the next
        int data_index = (step + ( 1 << (logN - 1 - stage))) & (size / 8 - 1);

        data = butterfly(data);

        if (complex_stage) {
            data = complex_rotate(data, data_index, stage, size);
        }

        data = swap(data);

        // Compute the delay of this stage
        int delay = 1 << (logN - 2 - stage);

        // Reordering multiplexers must toggle every 'delay' steps
        bool toggle = data_index & delay;

        // Base index of the reordered data
        unsigned head_index = size - (1 << (logN - stage + 2)) + 8 * (stage - 2);
        data = reorder_data(data, delay, fft_delay_elements, head_index, toggle);

        if (!complex_stage) {
            data = trivial_rotate(data);
        }
    }

    // Stage logN - 1
    data = butterfly(data);

    // Shift the contents of the sliding window. The hardware is capable of 
    // shifting the entire contents in parallel if the loop is unrolled. More
    // important, when unrolling this loop each transfer maps to a trivial 
    // loop-carried dependency
__attribute__((opencl_unroll_hint()))
    for (int ii = 0; ii < size + 8 * (logN - 2) - 1; ii++) {
        fft_delay_elements[ii] = fft_delay_elements[ii + 1];
    }

    if (inverse) {
       data = swap_complex(data);
    }

    return data;
}

