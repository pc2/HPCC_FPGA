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

#define DATA_TYPE long
#define DATA_TYPE_UNSIGNED unsigned DATA_TYPE

#ifndef UPDATE_SPLIT
#define UPDATE_SPLIT 1024
#endif

#define POLY 7


// SIMD not used, and instead CU replication since we have random accesses
__attribute__((num_simd_work_items(1)))
__attribute__((num_compute_units(UPDATE_SPLIT)))
__kernel
void accessMemory(__global volatile DATA_TYPE_UNSIGNED* restrict data,
                  __global const DATA_TYPE_UNSIGNED* restrict ran_const,
                  ulong m) {
    DATA_TYPE_UNSIGNED ran = ran_const[get_global_id(0)];

    uint mupdate = 4 * m;
    // do random accesses
    for (int i=0; i< mupdate / UPDATE_SPLIT; i++) {
        DATA_TYPE_UNSIGNED v = 0;
        if (((DATA_TYPE) ran) < 0) {
            v = POLY;
        }
        ran = (ran << 1) ^ v;
        DATA_TYPE_UNSIGNED address = ran & (m - 1);
        data[address] ^= ran;

    }
}
