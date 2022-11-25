========================
Execution of a Benchmark
========================

The host applications provided for every benchmark share the same code base. Thus, many input parameters are also shared between the benchmarks, which leads to a
very similar handling.
In this section, the input parameters that are shared between all benchmarks are explained in more detail.
Input parameters (or options) can be appended to the host execution call like this:

.. code-block:: bash

    STREAM_FPGA_intel -f PATH [...]

``-h``:
    When this flag is given, the benchmark will ignore all other input parameters and show a list of all available parameters.
    This can be useful to find out more about possible input options without the need to rely on additional documentation.
    So you can re-check, which input parameters are available to configure the execution of a given benchmark.
    Besides that, it will also show the default value for every parameter which will be used if it is not explicitly set and the version
    number of the benchmark. You can compare it with the `CHANGELOG` to check if a specific feature is available in the given version.

``-f, --file PATH``:
    This input parameter specifies the path to the bitstream that should be used to program the FPGA. It exists no default value for this parameter, so it always has to
    be given explicitly. Otherwise, the host application will abort.

``-n INT``:
    The number of repetitions can be given with this parameter as a positive integer. The benchmark experiment will be repeated the given number of times. The benchmark will show 
    the aggregated results for all runs, but only validate the output of the last run.

``--platform INT``:
    Also an integer. It can be used to specify the index of the OpenCL platform that should be used for execution. By default, it is set to -1. This will make the host code ask you
    to select a platform if multiple platforms are available. This option can become handy if you want to automize the execution of your benchmark.

``--device INT``:
    Also an integer. It can be used to specify the index of the OpenCL device that should be used for execution. By default, it is set to -1. This will make the host code ask you
    to select a device if multiple devices are available. This option can become handy if you want to automize the execution of your benchmark.

``-r INT``:
    A positive integer that specifies the number of kernel replications that are implemented in the bitstream given with `-f`. This allows to only use a subset of kernel replications 
    or different bitstreams with a varying number of kernel replications with the same host code. The options may not be available if the benchmark does not support kernel replication.

``--skip-validation``:
    In some cases, it may not be necessary to validate the result of the benchmark. For example, if different input parameters are tested, the input data is huge and validation takes a lot of time.
    Please note, that the benchmark will always fail with this option since it assumes the validation failed, so it will return a non-zero exit code! For reported measurements, the validation has to be enabled and the host should return
    with an exit code 0.

``--comm-type COMM``:
    This parameter chooses the communication strategy which will be used. Current Options are "IEC" for using the Intel External Channel, "PCIE" for using the host-to-host communicationa and "CPU" for calculating on the CPU.

``--test``:
    This option will also skip the execution of the benchmark. It can be used to test different data generation schemes or the benchmark summary before the actual execution. Please note, that the 
    host will exit with a non-zero exit code, because it will not be able to validate the output.

Additionally, every benchmark will have several options to define the size and type of the used input data.
These options vary between the benchmarks. An easy way to find out more about these options is to use the ``-h`` option with the host.