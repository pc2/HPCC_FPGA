/*
Copyright (c) 2022 Marius Meyer
          (c) 2024 Gerrit Pape

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

void recv_data(
    unsigned int num_iterations,
    unsigned int chunks,
    hls::stream<stream_word> &data_in,
    hls::stream<ap_uint<512>, 256> &data_stream,
    hls::stream<notify_word>& notify
) {
dump_iterations:
    for (unsigned int n = 0; n < num_iterations; n++) {
    dump_chunks:
        for (int i = 0; i < chunks; i++) {
#pragma HLS PIPELINE II = 1
            data_stream.write(data_in.read().data);
        }
        notify_word word;
        notify.write(word);
    }
}

void write_data(
    unsigned int num_iterations,
    unsigned int chunks,
    hls::stream<ap_uint<512>, 256> &data_stream,
    ap_uint<512> *write_buffer
) {
write_iterations:
    for (unsigned int n = 0; n < num_iterations; n++) {
    write_chunks:
        for (int i = 0; i < chunks; i++) {
#pragma HLS PIPELINE II = 1
            write_buffer[i] = data_stream.read();
        }
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

#pragma HLS dataflow
    int chunks = (size + 15) / 16;
    hls::stream<ap_uint<512>, 256> data_stream;

    recv_data(num_iterations, chunks, data_in, data_stream, notify);
    write_data(num_iterations, chunks, data_stream, write_buffer);
}

void read_data(
    unsigned int iterations,
    unsigned int chunks,
    ap_uint<512> *data_input,
    hls::stream<ap_uint<512>, 256> &data_stream
) {
read_iterations:
    for (unsigned int n = 0; n < iterations; n++) {
    read_chunks:
        for (unsigned int i = 0; i < chunks; i++) {
            #pragma HLS PIPELINE II = 1
            data_stream.write(data_input[i]);
        }
    }
}

void send_data(
    unsigned int num_iterations,
    unsigned int chunks,
    unsigned int max_frame_size,
    unsigned int dest,
    hls::stream<ap_uint<512>, 256> &data_stream,
    hls::stream<stream_word> &data_out,
    hls::stream<notify_word> &notify
) {
issue_iterations:
    for (unsigned int n = 0; n < num_iterations; n++) {
    issue_chunks:
        for (unsigned int i = 0; i < chunks; i++) {
            #pragma HLS PIPELINE II = 1
            stream_word word;
            word.data = data_stream.read();
            unsigned int count = i + 1;
            word.last = (count == chunks) || ((count % max_frame_size) == 0);
            if (word.last) {
                word.keep = -1;
            }
            word.dest = dest;
            data_out.write(word);
        }
        notify_word w = notify.read();
    }
}

void send_stream(ap_uint<512> *read_buffer, ap_uint<32> size, ap_uint<32> num_iterations, ap_uint<32> dest,
                 ap_uint<32> max_frame_size_log2, STREAM<stream_word> &data_out, STREAM<notify_word> &notify)
{
#pragma HLS INTERFACE m_axi port = read_buffer bundle = gmem_in
#pragma HLS INTERFACE s_axilite port = size
#pragma HLS INTERFACE s_axilite port = num_iterations
#pragma HLS INTERFACE s_axilite port = dest
#pragma HLS INTERFACE axis port = data_out
#pragma HLS INTERFACE s_axilite port = return

#pragma HLS dataflow
    unsigned int chunks = (size + 15) / 16;
    unsigned int max_frame_size = (1 << max_frame_size_log2);
    hls::stream<ap_uint<512>, 256> data_stream;

    read_data(num_iterations, chunks, read_buffer, data_stream);
    send_data(num_iterations, chunks, max_frame_size, dest, data_stream, data_out, notify);
}
