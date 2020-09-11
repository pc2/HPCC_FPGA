.. _beff: 
=======
b_eff
=======

This section contains all information related to the network bandwidth benchmark.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   */index


------------------------
Configuration Parameters
------------------------

In :numref:`beff_config` the configuration parameters are shown that are used to modify the kernel. 
All other parameters can also later be changed with the host code during runtime. 

*Note: The current implementation is optimized for circuit-switched network with four channels. Packet-switched networks might need a different kernel architecture. This specialization is the reason for the low amount of configuration parameters which might change, if more systems with inter-FPGA communication are available.*

.. _beff_config:
.. list-table:: Configuration parameters for the Kernel
   :widths: 15 25
   :header-rows: 1

   * - Parameter
     - Description
   * - ``CHANNEL_WIDTH``
     - Width of an external channel in bytes.


--------------------
Detailed Description
--------------------

The b_eff implementation for FPGA is based on the `effective bandwidth benchmark <https://fs.hlrs.de/projects/par/mpi//b_eff/>`_ that uses MPI to measure the network bandwidth using different message sizes.
The basic idea behind the benchmark is to combine both, the network latency and bandwidth, into a single metric: the effective bandwidth.
It is a metric that combines the network bandwidth measured for a wide range of message sizes to allow an estimation of how the network will react to different application scenarios.
The benchmark works mainly within a ring topology. Positions in the ring will be assigned at random.
For the use with a circuit-switched network, the benchmark rules were slightly adapted and simplified because routing does not take place in the network.
Instead, the communication links between the FPGAs are set up directly before execution in the required topology.

The benchmark works on 21 different message sizes :math:`L` that are defined like this:

.. math::
   L=1B,2B,4B,\dots,2kB,4kB,4kB*(a^1),4kB*(a^2),\dots,4kB*(a^8)

where :math:`a` is a integer constant that is set to 2 in this implementation.
So the final message sizes will be in the range :math:`2^0, 2^1, \dots, 2^{20}` bytes.
The maximum message size can be adjusted during runtime using the host code flag ``-m``.

Since we only use one topology the equation for the effective bandwidth shrinks to:

.. math::
    b_{eff} = sum_L(max_{rep}(b(L, rep))) / 21
    
where b(L, rep) is the measured bandwidth for a data size :math:`L` in repetition :math:`rep`.
    
The `looplength` described in the original benchmark will be used to minimize measurement 
errors caused by too short runtimes.
It specifies the total number of message exchanges that will be done with certain message size.
It may vary between message sizes to achieve similar runtimes independent of the message size.
In the current implementation the final loop length will be calculated based on an initial maximum loop length that can be given during runtime.
The number of repetitions for each message size is then calculated with the following equation:

.. math::
   max(\frac{u}{\lceil\frac{m}{2 \cdot CHANNEL\_WIDTH}\rceil}, l)

were :math:`u` is the upper/maximum loop length that should be used and :math:`l` the lowest. :math:`m` is the current message size in bytes
and :math:`CHANNEL\_WIDTH` the width of the communication channel in bytes.

The bandwidth will be calculated by measuring the execution time :math:`t_{exe}` of the kernel and dividing the measured time by the total number of transmitted bytes.

.. math::
    b(L) = \frac{L \cdot looplength}{t_{exe}}
    
The aggregated bandwidth is then calculated with:

.. math::
    b_{eff} = \frac{sum_L(b(L))}{21}


----------------------
Implementation Details
----------------------

The implementation consists of two different kernel types: a `send` kernel and a `receive` kernel.
During execution, they continuously send or receive data over two external channels.
Because the message size might exceed the width of the channels, the messages are further divided into data chunks that match the channel width.
Thus, a message is streamed chunk-wise over two channels to the receiver in a pipelined loop.
At kernel start, the `send` kernel will generate a message chunk that is filled with bytes of the value :math:`ld(m) mod 256`.
The message chunk will be used continuously for sending and will be stored in global memory after the last transmission.
This allows verifying the correct transmission of the data chunk over the whole range of repetitions.

A schematic view of the channel connections for the kernel implementation is given in :numref:`network_kernel_flow`.

.. _network_kernel_flow:
.. figure:: network_topology_setup.drawio.png
  :width: 360
  :align: center
  :alt: Connection of external channels to FPGA kernels

  The different kernels of the implementation are connected over bidirectional external channels to kernels running on other FPGAs. It this example two FPGAs form a bidirectional ring and will send and receive data from the other FPGA.



--------------------
Configuration Hints
--------------------

Configuration is currently not possible because of the still not fixed network properties.

