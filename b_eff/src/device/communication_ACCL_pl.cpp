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

extern "C" {

void send_recv(char *read_buffer,char *write_buffer,  unsigned int size_in_bytes, unsigned int num_iterations, 
                unsigned int neighbor_rank, addr_t communicator_addr,
                hls::stream<ap_uint<32> > &cmd, hls::stream<ap_uint<32> > &sts) {
    for (int i = 0; i < num_iterations; i++) {
        ACCLCommand accl_cmd(cmd, sts, communicator_addr, 0,0,0);
        accl_cmd.send(size_in_bytes, 0, neighbor_rank, read_buffer);
        accl_cmd.recv(size_in_bytes, 0, neighbor_rank, write_buffer);
    }
}
}