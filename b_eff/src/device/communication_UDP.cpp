/*
Copyright (c) 2022 Marius Meyer

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
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "parameters.h"

#include "etc/autopilot_ssdm_op.h"

#define STREAM hls::stream

typedef ap_axiu<512, 0, 0, 16> stream_word;
typedef ap_axiu<1, 0, 0, 0> notify_word;

void write_data(ap_uint<512> *read_buffer, ap_uint<32> size, ap_uint<32> max_frame_size, ap_uint<32> dest,
                STREAM<stream_word> &data_out)
{
    // receive the incoming data while send may still be in progress
    for (int chunk = 0; chunk < (size + 15) / 16; chunk++) {
#pragma HLS pipeline II = 1
        stream_word word;
        word.last = ((chunk + 1) == (size + 15) / 16) || (chunk > 0 && (chunk % (1 << max_frame_size) == 0));
        word.keep = -1;
        word.dest = dest;
        word.data = read_buffer[chunk];
        data_out.write(word);
    }
}

void read_data(ap_uint<512> *write_buffer, ap_uint<32> size, STREAM<stream_word> &data_in)
{
    // receive the incoming data while send may still be in progress
    for (int chunk = 0; chunk < (size + 15) / 16; chunk++) {
#pragma HLS pipeline II = 1
        stream_word word = data_in.read();
        write_buffer[chunk] = word.data;
    }
}

void recv_stream(ap_uint<512> *write_buffer, ap_uint<32> size, ap_uint<32> num_iterations, ap_uint<32> notify_enabled,
                 STREAM<stream_word> &data_in, STREAM<notify_word> &notify)
{
#pragma HLS INTERFACE m_axi port = write_buffer bundle = gmem_out
#pragma HLS INTERFACE s_axilite port = size
#pragma HLS INTERFACE s_axilite port = num_iterations
#pragma HLS INTERFACE s_axilite port = notify_enabled
#pragma HLS INTERFACE axis port = data_in
#pragma HLS INTERFACE axis port = notify
#pragma HLS INTERFACE s_axilite port = return

    notify_word w;
    for (int i = 0; i < num_iterations; i++) {
#pragma HLS protocol fixed
        read_data(write_buffer, size, data_in);
        ap_wait();
        if (notify_enabled != 0) {
            notify.write(w);
        }
    }
}

void send_stream(ap_uint<512> *read_buffer, ap_uint<32> size, ap_uint<32> num_iterations, ap_uint<32> dest,
                 ap_uint<32> max_frame_size, STREAM<stream_word> &data_out, STREAM<notify_word> &notify)
{
#pragma HLS INTERFACE m_axi port = read_buffer bundle = gmem_in
#pragma HLS INTERFACE s_axilite port = size
#pragma HLS INTERFACE s_axilite port = num_iterations
#pragma HLS INTERFACE s_axilite port = dest
#pragma HLS INTERFACE axis port = data_out
#pragma HLS INTERFACE s_axilite port = return

    for (int i = 0; i < num_iterations; i++) {
#pragma HLS protocol fixed
        write_data(read_buffer, size, dest, max_frame_size, data_out);
        ap_wait();
        notify_word w = notify.read();
    }
}
