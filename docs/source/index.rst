.. HPCC FPGA documentation master file, created by
   sphinx-quickstart on Thu Mar 26 10:51:40 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
HPCC FPGA Documentation
=============================================

In the last years, FPGAs with support for OpenCL-based high-level synthesis emerged as a new promising accelerator for high-performance computing applications.
This benchmarks suite uses the benchmarks defined in the HPC Challenge benchmark suite proposed in [LUS05]_ to implement parametrizable OpenCL kernels.
The parametrization of the kernels provided by the base implementation enables a better resource utilization on a broad range of Intel and Xilinx FPGA boards without manual changes in the source code.

----------------------
Benchmark Descriptions
----------------------

The following pages contain information about the base implementation for all benchmarks contained in the benchmark suite.

.. toctree::
   :maxdepth: 1
   :glob:

   STREAM/*
   RandomAccess/*
   FFT/*
   GEMM/*
   PTRANS/*
   LINPACK/*
   b_eff/*

-----------------
Technical Support
-----------------

This section collects topics that can be useful for the configuration, build and execution of the benchmarks.

.. toctree::
   :maxdepth: 1
   :glob:

   technical_support/*/*


-----------------
Benchmark Results
-----------------

These pages contain measurement results for the base implementations of the benchmarks. They are reported together with the used CPU and other relevant infrastructure, as well as the configuration and resource utilization of the bitsreams.


.. toctree::
   :maxdepth: 1
   :glob:

   */results/index


.. _overall_result_plot:
.. figure:: overall_results.png
  :height: 300
  :align: center
  :alt: Results can not be shown!

  Radar plot that contains the normalized results for the benchmarks STREAM, RandomAccess, FFT, and GEMM executed on three different boards and memory types.
   

.. [LUS05] LUSZCZEK, Piotr, et al. `Introduction to the HPC challenge benchmark suite`. Ernest Orlando Lawrence Berkeley NationalLaboratory, Berkeley, CA (US), 2005.


