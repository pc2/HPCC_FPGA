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
#include "accl_hls.h"


void send_recv(ap_uint<64> read_buffer,ap_uint<64> write_buffer,  ap_uint<32> size, ap_uint<32> num_iterations,
                ap_uint<32> neighbor_rank, ap_uint<32> communicator_addr, ap_uint<32> datapath_cfg,
                STREAM<command_word> &cmd, STREAM<command_word> &sts) {
#pragma HLS INTERFACE s_axilite port=read_buffer
#pragma HLS INTERFACE s_axilite port=write_buffer
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=num_iterations
#pragma HLS INTERFACE s_axilite port=neighbor_rank
#pragma HLS INTERFACE s_axilite port=communicator_addr
#pragma HLS INTERFACE s_axilite port=datapath_cfg
#pragma HLS INTERFACE axis port=cmd
#pragma HLS INTERFACE axis port=sts
#pragma HLS INTERFACE s_axilite port=return
    accl_hls::ACCLCommand accl(cmd, sts);
    for (int i = 0; i < num_iterations; i++) {
        #pragma HLS protocol fixed
        accl.start_call(
            ACCL_SEND, size, communicator_addr, neighbor_rank, 0, 0,
                datapath_cfg, 0, 0,
                read_buffer, 0, 0);
        ap_wait();
        accl.finalize_call();
        ap_wait();
        accl.start_call(
            ACCL_RECV, size, communicator_addr, neighbor_rank, 0, 0,
                datapath_cfg, 0, 0,
                0, 0, write_buffer);
        ap_wait();
        accl.finalize_call();
    }
}

