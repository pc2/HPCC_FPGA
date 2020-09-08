========================
Multi-FPGA Benchmarking
========================

The benchmarks of the HPCC FPGA benchmark suite support the execution on multiple FPGAs.
It tries to keep the execution model close to the one of the HPCC benchmark for CPU which distinguishes between `global` and `embarrassingly parallel` execution.
The difference is, that for the global execution, the calculation of a single problem done with all available resources.
So the FPGAs will need to communicate and exchange data during calculation.
On the other hand, with an embarrassingly parallel execution, every FPGA works on its own data, so no communication is needed during calculation.

The benchmarks can be divided as follows:

Global:
    - :ref:`RandomAccess <randomaccess>`
    - :ref:`FFT <fft>`
    - :ref:`PTRANS <ptrans>`
    - :ref:`LINPACK <hpl>`
    - :ref:`b_eff <beff>`

Embarrassingly Parallel:
    - :ref:`STREAM <stream>`
    - :ref:`GEMM <gemm>`

Independent of the used paralleization scheme, the benchmarks configuration of the benchmarks is nearly the same as described in the next section.

-----------------------------------
Execution of a multi-FPGA benchmark
-----------------------------------

A host process of the benchmark will always only handle a single FPGA device.
Nevertheless, multiple host processes can communicate over MPI to execute a benchmark in parallel and synchronize the data.
If multiple FPGAs are available on a single host, they can be utilized by executing multiple MPI ranks on this host.
The FPGAs will be assigned to the MPI ranks in a round robin scheme so the device ID for every MPI rank is calculated with :math:`id = mpi\_rank\ mod\ \#devices`.

As an example we execute STREAM with two MPI ranks and two kernel replications each:

.. code:: Bash

    mpirun -n 2 .STREAM_FPGA_intel -f kernel.aocx -r 2 -s 1024

This will execute STREAM on each FPGA with array sizes of 1024 elements.
These 1024 elements will be distributed amongst the two kernel replication such that every kernel replication will work on arrays of size 512.
Since the execution includes two MPI ranks and the input sizes will be scaled by the number of ranls, the total size of the arrays will be :math:`1024 \cdot 2 = 2048`.
This also holds for all other benchmarks, so the basic rule of thumb is: 

*The data size specified in the benchmark configuration will be multiplied by the number of MPI ranks and and then equally distribued to the total available kernel replications*

