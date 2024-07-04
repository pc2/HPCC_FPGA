====================================
GEMM FPGA Benchmark Results
====================================

The benchmark results are given divided by the used version of the benchmark, since internal changes in the benchmark code might lead to different performance results.
All measurements were done with single precision floating point matrices of size 4096x4096 which equals 64 MB of data.
If this size was not evenly dividable by the number of replications, the matrix size was further reduced to achieve equal load for every kernel replication.
The measurements were executed 10 times and the best result is published.

The results and the used configuration is given in :numref:`tbl_gemm_1_0_results` and are also available as :download:`CSV <gemm-1-0.csv>`.

.. _tbl_gemm_1_0_results:
.. csv-table:: GEMM FPGA Benchmark Results
    :file: gemm-1-0.csv
    :stub-columns: 1

