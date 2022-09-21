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


void send_recv(float *read_buffer,float *write_buffer,  ap_uint<32> size, ap_uint<32> num_iterations, 
                ap_uint<32> neighbor_rank, ap_uint<32> communicator_addr, ap_uint<32> datapath_cfg,
                hls::stream<command_word> &cmd, hls::stream<command_word > &sts) {
    accl_hls::ACCLCommand accl_cmd(cmd, sts, communicator_addr, datapath_cfg,0,0);
    for (int i = 0; i < num_iterations; i++) {
        accl_cmd.send(size, 0, neighbor_rank, (ap_uint<64>)read_buffer);
        accl_cmd.recv(size, 0, neighbor_rank, (ap_uint<64>)write_buffer);
    }
}
