.. HPCC FPGA documentation master file, created by
   sphinx-quickstart on Thu Mar 26 10:51:40 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================================
HPCC FPGA Documentation
=============================================

This repository contains a more detailed documentation of the HPC Challenge Benchmark for FPGA.
In contrast to the code documentation maintained using Doxygen, this documentation focuses on a broader picture for all benchmarks.
This includes a more detailed description of the benchmark implementations, the configuration options and their impact on performance 
and resource usage, and performance results for selected FPGAs with the used configurations.

.. toctree::
   :maxdepth: 1
   :caption: Benchmark Descriptions:
   :glob:

   STREAM/*
   RandomAccess/*
   FFT/*
   GEMM/*
   PTRANS/*
   LINPACK/*
   b_eff/*

.. toctree::
   :maxdepth: 1
   :caption: Other Topics:
   :glob:

   technical_support/*/*

.. toctree::
   :maxdepth: 1
   :caption: Benchmark Results:
   :glob:

   */results/index
   

Or use the :ref:`search`.
