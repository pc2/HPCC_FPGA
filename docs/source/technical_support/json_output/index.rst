===========
JSON Output
===========

The output of the configuration, settings, timings and results in machine-readable json-format can be enabled as described in :ref:`Execution of a Benchmark <execution>`

When enabled, this creates a json file which will have some information for all benchmarks. In the following example the different informations are left out, so these are the same for all benchmarks.

.. code-block:: javascript

    {
      "config_time": "Mon Dec 05 15:09:08 UTC 2022",
      "device": "Intel(R) FPGA Emulation Device",
      "environment": {
        "LD_LIBRARY_PATH": "/opt/software/pc2/EB-SW/software/Python/3.9.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libffi/3.3-GCCcore-10.3.0/lib64:/opt/software/pc2/EB-SW/software/GMP/6.2.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/SQLite/3.35.4-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/Tcl/8.6.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libreadline/8.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libarchive/3.5.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/cURL/7.76.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/bzip2/1.0.8-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ncurses/6.2-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/ScaLAPACK/2.1.0-gompi-2021a-fb/lib:/opt/software/pc2/EB-SW/software/FFTW/3.3.9-gompi-2021a/lib:/opt/software/pc2/EB-SW/software/FlexiBLAS/3.0.4-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.15-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenMPI/4.1.1-GCC-10.3.0/lib:/opt/software/pc2/EB-SW/software/PMIx/3.2.3-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libfabric/1.12.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/UCX/1.10.0-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libevent/2.1.12-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/OpenSSL/1.1/lib:/opt/software/pc2/EB-SW/software/hwloc/2.4.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libpciaccess/0.16-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/libxml2/2.9.10-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/XZ/5.2.5-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/numactl/2.0.14-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/binutils/2.36.1-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/zlib/1.2.11-GCCcore-10.3.0/lib:/opt/software/pc2/EB-SW/software/GCCcore/10.3.0/lib64:/opt/software/slurm/21.08.6/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/21.2.0/hld/host/linux64/lib:/opt/software/FPGA/IntelFPGA/opencl_sdk/20.4.0/hld/board/bittware_pcie/s10/linux64/lib"
      },
      "git_commit": "c7f3890-dirty",
      "mpi": {
        "subversion": 1,
        "version": 3
      },
      "name": "effective bandwidth",
      "results": {
      },
      "settings": {
        "Communication Type": "IEC",
        "Kernel File": "./communication_bw520n_IEC_emulate.aocx",
        "Kernel Replications": 2,
        "MPI Ranks": 1,
        "Repetitions": 10,
        "Test Mode": "No"
      },
      "timings": {
      },
      "version": "1.3"
    }

If a benchmark has more settings, they will be added to the settings-key. Every benchmark can track different categories of timings, different results and errors. To see a full example and which keys are available have a look at the README.md of the single benchmarks in the [git repositoy](https://git.uni-paderborn.de/pc2/HPCC_FPGA).

The results and timings are in a special format, which consists of the value and the unit.

.. code-block:: javascript

    {
      "results": {
        "b_eff": {
          "unit": "B/s",
          "value": 14806691.755972749
        }
      }
    }

The timings are a vector of all the timings which were measured, expect for b_eff, where a special format is used. For every message size used in the benchmark the interim results are saved in the following way.

.. code-block:: javascript

    {
    "6": {
      "maxCalcBW": 9225059.007945802,
      "maxMinCalculationTime": 5.5501e-05,
      "timings": [
        {
          "looplength": 4,
          "messageSize": 6,
          "timings": [
            {
              "unit": "s",
              "value": 0.008889638
            },
            {
              "unit": "s",
              "value": 0.000115271
            },
            {
              "unit": "s",
              "value": 0.000149272
            },
            {
              "unit": "s",
              "value": 0.000163372
            },
            {
              "unit": "s",
              "value": 7.5731e-05
            },
            {
              "unit": "s",
              "value": 5.5501e-05
            },
            {
              "unit": "s",
              "value": 0.000162132
            },
            {
              "unit": "s",
              "value": 8.2091e-05
            },
            {
              "unit": "s",
              "value": 6.7621e-05
            },
            {
              "unit": "s",
              "value": 0.000126891
            }
          ]
        }
      ]
    },
    "7": {
      "maxCalcBW": 12222341.581026724,
      "maxMinCalculationTime": 8.3781e-05,
      "timings": [
        {
          "looplength": 4,
          "messageSize": 7,
          "timings": [
            {
              "unit": "s",
              "value": 0.000296573
            },
            {
              "unit": "s",
              "value": 0.000136292
            },
            {
              "unit": "s",
              "value": 0.000320834
            },
            {
              "unit": "s",
              "value": 0.000130881
            },
            {
              "unit": "s",
              "value": 8.3781e-05
            },
            {
              "unit": "s",
              "value": 0.000247252
            },
            {
              "unit": "s",
              "value": 0.000430356
            },
            {
              "unit": "s",
              "value": 0.000281403
            },
            {
              "unit": "s",
              "value": 0.000421565
            },
            {
              "unit": "s",
              "value": 0.000266754
            }
          ]
        }
      ]
    },
    "8": {
      "maxCalcBW": 38030862.93662141,
      "maxMinCalculationTime": 5.3851e-05,
      "timings": [
        {
          "looplength": 4,
          "messageSize": 8,
          "timings": [
            {
              "unit": "s",
              "value": 0.000157722
            },
            {
              "unit": "s",
              "value": 0.000121611
            },
            {
              "unit": "s",
              "value": 0.000217192
            },
            {
              "unit": "s",
              "value": 9.7101e-05
            },
            {
              "unit": "s",
              "value": 6.6931e-05
            },
            {
              "unit": "s",
              "value": 8.6791e-05
            },
            {
              "unit": "s",
              "value": 0.000145572
            },
            {
              "unit": "s",
              "value": 0.000143042
            },
            {
              "unit": "s",
              "value": 8.5281e-05
            },
            {
              "unit": "s",
              "value": 5.3851e-05
            }
          ]
        }
      ]
    }
    }
