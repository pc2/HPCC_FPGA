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

/**
 * This file contains two kernels "send" and "recv".
 * They are identical in their function except two differences:
 * - they use different channels for sending and receiving data
 * - "send" sends first a message and then receives,  "recv" does it the other way around
 *
 * The kernels are hardcoded to work with the Bittware 520N board that offers
 * 4 external channels.
 * The file might need to be adapted for the use with other boards!
 */


#include "parameters.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

#define ITEMS_PER_CHANNEL (CHANNEL_WIDTH / sizeof(DEVICE_DATA_TYPE))

/**
 * Data type used for the channels
 * Consider using vector types instead
 */
typedef struct {
    DEVICE_DATA_TYPE values[ITEMS_PER_CHANNEL];
} message_part;

/**
 * Definition of the external channels
 */
 // PY_CODE_GEN block_start [replace(local_variables=locals()) for r in range(num_replications)]
channel message_part ch_out_/*PY_CODE_GEN  2*r+1*/ __attribute((io(/*PY_CODE_GEN "\"kernel_output_ch" + str(r % 4) + "\""*/)));
channel message_part ch_out_/*PY_CODE_GEN  2*r+2*/  __attribute((io(/*PY_CODE_GEN "\"kernel_output_ch" + str((r + 2) % 4) + "\""*/)));
channel message_part ch_in_/*PY_CODE_GEN  2*r+1*/ __attribute((io(/*PY_CODE_GEN "\"kernel_input_ch" + str(r % 4) + "\""*/)));
channel message_part ch_in_/*PY_CODE_GEN  2*r+2*/  __attribute((io(/*PY_CODE_GEN "\"kernel_input_ch" + str((r + 2) % 4) + "\""*/)));
channel message_part ch_exchange/*PY_CODE_GEN 2*r+1*/;
channel message_part ch_exchange/*PY_CODE_GEN 2*r+2*/;
// PY_CODE_GEN block_end


// PY_CODE_GEN block_start [replace(local_variables=locals()) for r in range(num_replications)]
/**
 * Send kernel that will send messages through two channels
 *
 * @param data_size Size of the used message
 * @param repetitions Number of times the message will be sent and received
 */
__kernel
__attribute__ ((max_global_work_dim(0)))
void send/*PY_CODE_GEN  r*/(const unsigned data_size,
        const unsigned repetitions) {
    const unsigned send_iterations = ((1 << data_size) +  2 * ITEMS_PER_CHANNEL - 1) / (2 * ITEMS_PER_CHANNEL);
    message_part send_part1;
    message_part send_part2;

    // Initialize the data chunks that are sent over the external channels
    __attribute__((opencl_unroll_hint(ITEMS_PER_CHANNEL)))
    for (DEVICE_DATA_TYPE d = 0; d < ITEMS_PER_CHANNEL; d++) {
        send_part1.values[d] = data_size & 255;
        send_part2.values[d] = data_size & 255;
    }

    // Sent a message multiple times over the external channels
    for (unsigned i=0; i < repetitions; i++) {
        // Send a single message sent over two channels split into multiple chunks
        for (unsigned k=0; k < send_iterations; k++) {
            write_channel_intel(ch_out_/*PY_CODE_GEN  2*r+1*/, send_part1);
            write_channel_intel(ch_out_/*PY_CODE_GEN  2*r+2*/, send_part2);
        }
#ifndef EMULATE
        // Introduce data dependency between loop iterations to prevent coalescing of loop
        send_part1 = read_channel_intel(ch_exchange/*PY_CODE_GEN 2*r+1*/);
        send_part2 = read_channel_intel(ch_exchange/*PY_CODE_GEN 2*r+2*/);
#endif
    }
}


/**
 * Receive kernel that will send messages through channel 3 and 4 and receive messages from
 * channel 3 and 4 in parallel.
 *
 * @param data_size Size of the used message
 * @param repetitions Number of times the message will be sent and received
 */
__kernel
__attribute__ ((max_global_work_dim(0)))
void recv/*PY_CODE_GEN  r*/(__global DEVICE_DATA_TYPE* validation_buffer,
            const unsigned data_size,
            const unsigned repetitions) {
    const unsigned send_iterations = ((1 << data_size) +  2 * ITEMS_PER_CHANNEL - 1) / (2 * ITEMS_PER_CHANNEL);
    message_part recv_part1;
    message_part recv_part2;

    // Receive a message multiple times over the external channels
    for (unsigned i=0; i < repetitions; i++) {
        // Receive a single message sent over two channels split into multiple chunks
        for (unsigned k=0; k < send_iterations; k++) {
            recv_part1 = read_channel_intel(ch_in_/*PY_CODE_GEN  2*r+1*/);
            recv_part2 = read_channel_intel(ch_in_/*PY_CODE_GEN  2*r+2*/);
        }
#ifndef EMULATE
        // Introduce data dependency between loop iterations to prevent coalescing of loop
        // by sending the data to the send kernel
        write_channel_intel(ch_exchange/*PY_CODE_GEN 2*r+1*/, recv_part1);
        write_channel_intel(ch_exchange/*PY_CODE_GEN 2*r+2*/, recv_part2);
#endif
    }

    // Store the last received data chunks in global memory for later validation
    __attribute__((opencl_unroll_hint(ITEMS_PER_CHANNEL)))
    for (DEVICE_DATA_TYPE d = 0; d < ITEMS_PER_CHANNEL; d++) {
        validation_buffer[d] = recv_part1.values[d];
    }
    __attribute__((opencl_unroll_hint(ITEMS_PER_CHANNEL)))
    for (DEVICE_DATA_TYPE d = 0; d < ITEMS_PER_CHANNEL; d++) {
        validation_buffer[ITEMS_PER_CHANNEL + d] = recv_part2.values[d];
    }
}

//PY_CODE_GEN block_end
