.. _gemm:

======
GEMM
======

This section contains all information related to the GEMM benchmark.
This benchmark calculates the result for :math:`C' = \beta \cdot C + \alpha \cdot A \cdot B` where :math:`A,B,C,C' \in \Bbb R^{n \times n}` and :math:`\alpha, \beta \in \Bbb R`.
The kernel uses a two-leveled blocked approach that can be individually scaled to the target hardware.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   */index

------------------------
Configuration Parameters
------------------------

In :numref:`gemm_config` the configuration parameters are shown that are used to modify the kernel. 
All other parameters can also later be changed with the host code during runtime. 

.. _gemm_config:
.. list-table:: Configuration parameters for the Kernel
   :widths: 15 25
   :header-rows: 1

   * - Parameter
     - Description
   * - ``NUM_REPLICATIONS``
     - Replicates all kernels the given number of times. This allows to simultaneously schedule multiple kernels with different input data. The data set might be split between the available kernels to speed up execution.
   * - ``BLOCK_SIZE``
     - Size of the matrix blocks that are buffered in local memory.
   * - ``GEMM_SIZE``
     - Size of the smaller matrix blocks that are multiplied in registers.
   * - ``GLOBAL_MEM_UNROLL``
     - Unrolling factor for the loops that read or write to global memory. This modifies the width of the load and store units to optimize memory accesses.
   * - ``INTEL_MUL_SHIFT_REG``
     - This parameters specifies the size of the shift register that is used within the multiplication pipeline. This allows to relax memory dependencies and increase the final kernel frequency. If the value is set to 0, no shift register will be used. This parameter is only relevant for Intel devices.
   * - ``DATA_TYPE``
     - Specifies the used data type for the calculation. ``half``, ``float`` and ``double`` are supported.

--------------------
Detailed Description
--------------------

The GEMM benchmark implements a matrix-matrix multiplication similar to the GEMM routines in the BLAS library.
It calculates :math:`C' = \beta \cdot C + \alpha \cdot A \cdot B` where :math:`A,B,C,C' \in \Bbb R^{n \times n}` and :math:`\alpha, \beta \in \Bbb R`.
The number of FLOP for the performance calculation is defined to be :math:`2 \cdot n^3`.
The result is verified by calculating the residual :math:`\frac{||C - C'||}{\epsilon \cdot n \cdot ||C||_F}` where :math:`\epsilon` is the machine epsilon and :math:`C'` the result of the reference implementation.
The implementation is based on a matrix multiplication design for Intel Stratix 10 and generalized to make it compatible with a broader range of devices.
The kernel creates a memory hierarchy by doing the following data movements:

- A blocked matrix multiplication in global memory over a matrix of size n
- A blocked matrix multiplication in local memory (BRAM) over blocks of size ``BLOCK_SIZE``
- A fully unrolled matrix multiplication in registers over blocks of size ``GEMM_SIZE``


The data movements and used pipelines are shown in the :numref:`memory` .

.. _memory:
.. figure:: kernel_memory_hierarchy.drawio.png
  :width: 600
  :align: center
  :alt: Impact of parameters to split the matix into blocks

  Visualized data movements for the multiplication of two matrices A and B. These movements are implemented in two pipelines that will run sequentially.

The matrices are divided into blocks of a fixed size that can be modified with the ``BLOCK_SIZE`` parameter.
One pipeline loads a single block of matrix A and B into the local memory of the FPGA.
In the second pipeline the loaded blocks are used to calculate an intermediate result again with a blocked approach.
This time the block size is defined by the parameter ``GEMM_SIZE``.
The matrix multiplication for this block will be fully unrolled which allows to initiate the multiplication of a whole block every clock cycle.
Both pipelines will be executed sequentially to calculate a result of a single block of A + B.
A third pipeline loads a block of C and calculates the final result for a block of C'.

---------------------
Expected Bottlenecks
---------------------

Matrix multiplications are very compute-intensive, so the benchmark is expected to be mostly computation-bound.
Nevertheless, the two main pipelines that are executed sequentially are both, memory-bound (load data from global memory) and compute-bound (calculate on data).
The total execution time of the first pipeline is -- depending on the chosen block size -- considerably smaller than for the calculation pipeline.
So the benchmark will in the end be affected by both, the memory bandwidth and the calculation performance of the unrolled multiplication.
The calculation performance again is highly depending on the used FPGA resource and the final kernel frequency.
In consequence, the benchmark performance is also depending on the route and place capabilities of the development tools.

Together with the third pipeline, that is used to write the final result of a block back to global memory, the execution time can be modelled as given in :eq:`eq_gemm_performance`.

.. math::
    t_{exe} = \frac{b^2}{u \cdot f_{mem}} + \frac{b^3}{g^3 \cdot f_k} + \frac{b^2}{u \cdot \frac{n}{b} \cdot f_{mem}}
    :label: eq_gemm_performance

where 
  :math:`b` equals ``BLOCK_SIZE``,
  :math:`r` equals ``NUM_REPLICATIONS``,
  :math:`u` equals ``GLOBAL_MEM_UNROLL`` and
  :math:`g` equals ``GEMM_SIZE``

Moreover, :math:`n` is the total matrix size, :math:`f_k` the kernel frequency and :math:`f_{mem}` the frequency of the memory interface.

--------------------
Configuration Hints
--------------------

The benchmark is mainly calculation bound.
The most stressed resources are DSPs and BRAM, since they are needed to increase parallelism of the calculation and the pipeline depth.
A larger local memory buffer leads to lesser reads and writes to global memory and thus reduces the overhead these operations are introducing.
In general, the parameters for the GEMM benchmark should be choosen after the following criteria:

1. Choose the data type with the ``DATA_TYPE`` parameter.
2. Scale ``GEMM_SIZE`` to the largest power of two that fits onto the FPGA since this parameters sets the actual amount of calculations that will be done in parallel. The size might depend on the chosen data type, since DSPs might not be usable for all precisions.
3. Set ``BLOCK_SIZE`` to the largest power of two that fits into the local memory of the FPGA. This will lead to a better utilization of the calculation pipeline.
4. The ``GLOBAL_MEM_UNROLL`` parameter should be set to match the optimal width of the memory interface depending on the used data type. Check the reports how the load pipeline will be synthesized. In some cases it might be better to set ``GLOBAL_MEM_UNROLL`` to the same value as ``GEMM_SIZE`` since there will be no performance gain for higher values.
5. For Intel devices, check the estimated kernel frequency in the report. If it is too low or the calculation pipeline shows an II larger than 1, the shift register should be used. Choose a value larger 0 for ``INTEL_MUL_SHIFT_REG`` until frequency and II are in an acceptable range.
6. Replicate the kernel using the ``NUM_REPLICATIONS`` parameter to fill the FPGA. It might be better to pick smaller block sizes and instead replicate the kernel to make placing and routing easier.