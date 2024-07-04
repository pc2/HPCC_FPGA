.. _randomaccess:

============
RandomAccess
============

This section contains all information related to the RandomAccess benchmark.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   */index


------------------------
Configuration Parameters
------------------------

In :numref:`ra_config` the configuration parameters are shown that are used to modify the kernel. 
All other parameters can also later be changed with the host code during runtime. 

.. _ra_config:
.. list-table:: Configuration parameters for the Kernel
   :widths: 15 25
   :header-rows: 1

   * - Parameter
     - Description
   * - ``NUM_REPLICATIONS``
     - Replicates all kernels the given number of times. This allows to simultaneously schedule multiple kernels with different input data. The data set might be split between the available kernels to speed up execution.
   * - ``HPCC_FPGA_RA_DEVICE_BUFFER_SIZE``
     - Size of the local memory buffer that stores intermediate values read from global memory. Used to reduce the impact of memory latency on the execution time.
   * - ``HPCC_FPGA_RA_INTEL_USE_PRAGMA_IVDEP``
     - Use the pragma ``ivdep`` on the outer loop of the kernel implementation. This will tell the compiler to ignore the dependencies between read and write to the global memory and allows the implementation in a single pipeline (use together with ``DEVICE_BUFFER_SIZE=1``. NOTE: This might lead to a increased error above the allowed threshold!)
   * - ``HPCC_FPGA_RA_RNG_COUNT_LOG``
     - Log2 of the number of random number generators that will be generated per replication. This allows to achieve a better scaling behavior for large number of replications.
   * - ``HPCC_FPGA_RA_RNG_DISTANCE``
     - Distance between RNGs in shift register. Used to relax timing and increase clock frequency.


--------------------
Detailed Description
--------------------

The random access benchmark measures the performance for non-consecutive memory accesses expressed in the performance metric Mega Update Operations per Second (MUOPs).
It updates values in a data array :math:`d \in \Bbb Z^n` such that :math:`d_{i} = d_{i} \oplus a` where :math:`a \in \Bbb Z` is a value from a pseudo random sequence.
n is defined to be a power of two. 
As data type, 64 bit integers are used.
Since an exclusive or operation is used to update the values, applying the update again will lead to the initial value.
This is also used to verify the results on the host side.
The same updates are applied to the data array such that the data array should contain the initial values again, if all updates where successful.
The incorrect items are counted, and the error percentage is calculated with :math:`\frac{error}{n} \cdot 100`.
An error of :math:`<1\%` has to be accomplished to pass the validation. Hence, update errors caused by concurrent data accesses are tolerated to some degree.
The performance of the implementation is mainly bound by the memory bandwidth and latency. So the kernel should be replicated to utilize all available memory banks.

The benchmark requires :math:`4 \cdot n` updates on the array of length n.
Therefore, it uses a pseudo random sequence generator that is implemented in every kernel replication.
The index in the data array as well as the value of the update will be extracted from the random number.
For a more detailed description of the used pseudo random number scheme refer to the description on the `HPC Challenge website <https://icl.utk.edu/projectsfiles/hpcc/RandomAccess/>`_.
Since every replication of the kernel is in charge of a distinct range of addresses that is placed in their memory bank, the kernel has to check if the calculated address is actually within its range.
Only then the value will be loaded from global memory and stored in a local memory buffer which size can be configured over the ``HPCC_FPGA_RA_DEVICE_BUFFER_SIZE`` parameter.
When the buffer is full, the benchmark will write back the updated values from the buffer to global memory.
With this approach, read after write dependencies will be relaxed which prevents pipeline stalls caused by pending load operations.
However, the buffer introduces update errors, if the same value is loaded multiple times since there are no additional checks implemented.
Since the benchmark allows a small amount of errors, changing the buffer size can be used as a trade-off between benchmark error and performance.
Also it has to be noted that the memory accesses are following a pseudo random order.
This means that the measured memory bandwidth will be considerably lower than for sequential memory accesses.
One cause are the missing memory bursts which usually increase the efficiency.
Moreover, the values are most likely located in different pages of the DDR memory which will introduce additional overheads.

The parameter ``HPCC_FPGA_RA_RNG_COUNT_LOG`` allows to increase the amount of pseudo-random number generators per replications.
This increases the probability, that a valid index for every kernel replication will be generated in every clock cycle.
However, this also increases the logic usage and may lead to lower kernel frequencies.
On systems with many separate global memory banks like HBM boards or multi-FPGA setups where a lot of kernel replications are required, increasing the amount of RNGs
can increase the performance since the benchmark may get calculation bound by the RNGs otherwise.

--------------------
Configuration Hints
--------------------

The random access benchmark is not only limited by the random access global memory bandwidth but also by the calculation of the pseudo random number sequence.
Every kernel replication needs to calculate all random numbers in the sequence to check if they are in their range of responsibility.
For a high number of replications this will have a considerable impact in the measured performance because the number of updates per calculated random number will decrease.
Nevetheless, the number of replications should still be matched with the number of available memory banks to get the best performance.
``HPCC_FPGA_RA_DEVICE_BUFFER_SIZE`` has to be picked as large as possible to reduce the impact of the memory latency.
It should not be too large to not exceed the allowed error of 1% or reduce the kernel frequency.
Additionally, the amount of pseudo-random number generators should be increased for high numbers of kernel replications while keeping an eye on the kernel frequency.
Choosing values for ``HPCC_FPGA_RA_RNG_COUNT_LOG`` which are higher than log2(``NUM_REPLICATIONS``) will not give much performance benefits, because a single kernel replication can
only process a single address per clock cycle.

