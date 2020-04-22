# HPCC FPGA

HPCC FPGA is an OpenCL-based FPGA benchmark suite with focus on high performance computing.
It is based on the benchmarks of the well-established CPU benchmark suite HPCC.
This repository contains the OpenCL kernels and host code for all benchmarks together with the build scripts and instructions.

The repository is structured as follows:

- The code and build instructions for every benchmark is located in the subfolders
- Every benchmark comes with a separate host code and build scripts
- The repository contains multiple submodules used by the benchmarks contained in the `extern` folder. Make sure to initialize and update the submodules before building a benchmark with `git submodule update --init --recursive`!
