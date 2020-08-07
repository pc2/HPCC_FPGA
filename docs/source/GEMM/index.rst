======
GEMM
======

This section contains all information related to the GEMM benchmark.
This benchmark calculates the result for :math:`C' = \beta * C + \alpha * A * B` where :math:`A,B,C,C' \in \Bbb R^{n \times n}` and :math:`\alpha, \beta \in \Bbb R`.
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
It calculates :math:`C' = \beta * C + \alpha * A * B` where :math:`A,B,C,C' \in \Bbb R^{n \times n}` and :math:`\alpha, \beta \in \Bbb R`.
The number of FLOP for the performance calculation is defined to be :math:`2 * n^3`.
The result is verified by calculating the residual :math:`\frac{||C - C'||}{\epsilon n ||C||_F}` where :math:`\epsilon` is the machine epsilon and :math:`C'` the result of the reference implementation.
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