# Performance of the Single-Work-Item Kernel

In the following the performance of the SWI cannon kernel is discussed.

## Algorithm

The kernel is structured into three levels.

1. **register_gemm**: Multiplies two matrices fully unrolled in FPGA registers. Size of these matrices is defined by the build variable `GEMM_BLOCK`.
2. **local_mem_gemm**: Multiplies two matrices of size `BLOCK_SIZE` fully pipelined in local memory and scales the result. Uses internally **register_gemm**.
3. **gemm**: The first level multiplies two matrices that have a size multiple of `BLOCK_SIZE` in global memory and adds the result to a third matrix. Uses **local_gemm** for the multiplication.

The host code will call **gemm** and the compute performance is depending on the size of `GEMM_BLOCK`, since only there the actual computation is done.


## Performance Model

The calculation of a single block of the result matrix is fully pipelined.
Matrix multiplication and writeback to global memory are executed sequentially to reduce the memory replications and make it possible to store bigger matrices in local memory.

For the calculation of the runtime we have to look at both.

#### Theoretical Performance

To multiply two matrix blocks they first have to be loaded from global memory and and then processed by the local memory GEMM.
The load operations can be calculated with the following formula not considering the latency:

![t_{mempipeline}=\frac{s_{block}*s_{block}}{s_{bus}}](https://latex.codecogs.com/gif.latex?t_{mempipeline}=\frac{\frac{\(s_{block}\)^2}{s_{bus}}}{f}) 

where ![s_{block}](https://latex.codecogs.com/gif.latex?s_{block}) is the size of a row of the matrix block that has to be stored in local memory in bytes and 
![s_{bus}](https://latex.codecogs.com/gif.latex?s_{bus}) the bus width of the global memory controller in bytes.
![f](https://latex.codecogs.com/gif.latex?f) is the kernel frequency.
Moreover latency will be added to this operation for every DRAM page that has to be activated:

![t_{lat}=\frac{\(s_{block}\)^2}{s_{page}}*\(t_{RCD}+t_{RP}\)](https://latex.codecogs.com/gif.latex?t_{lat}=\frac{\(s_{block}\)^2}{s_{page}}*\(t_{RCD}+t_{RP}\))

where ![s_{page}](https://latex.codecogs.com/gif.latex?s_{page}) is the size of a DRAM page in bytes.
![t_{RCD}](https://latex.codecogs.com/gif.latex?t_{RCD}) and ![t_{RP}](https://latex.codecogs.com/gif.latex?t_{RP}) are the
row address to column address delay and the row precharge time of the used DRAM.

So the total time for the memory accesses for a single block is:

![t_{mem}=t_{mempipeline}+t_{lat}](https://latex.codecogs.com/gif.latex?t_{mem}=t_{mempipeline}+t_{lat}) 


For the calculation we have to consider the blocked matrix multiplication from global to local memory as well as the blocked matrix multiplication from local memory to registers.
The calculation within the registers is fully unrolled and will only add a constant latency to the calculation:

![t_{calc}=\frac{\(\frac{s_{block}}{s_{reg}}\)^3}{f}](https://latex.codecogs.com/gif.latex?t_{calc}=\frac{\(\frac{s_{block}}{s_{reg}}\)^3}{f}) 

where ![s_{reg}](https://latex.codecogs.com/gif.latex?s_{reg}) is the row of a register gemm block in bytes.

If ![t_{mem}<t_{calc}](https://latex.codecogs.com/gif.latex?t_{mem}<t_{calc}) then the calculation is compute bound.
The execution time for the whole matrix will then be:

![t_{total}=\(\frac{s_{total}}{s_{block}}\)^3*\(t_{calc}\)+\frac{s_{total}}{s_{block}}\)^3*t_{mem}](https://latex.codecogs.com/gif.latex?t_{total}=\(\frac{s_{total}}{s_{block}}\)^3*t_{calc}+\(\frac{s_{total}}{s_{block}}\)^2*t_{mem}) 

where ![s_{total}](https://latex.codecogs.com/gif.latex?s_{total}) is the size of a row of the whole matrix in bytes.

The first part of the equation represents the time that is needed for calculation.
It will be t_calc plus the latency of the pipeline.
We assume matrix size big enough to hide the latency of the calculation pipeline.
Thus it is not considered in the model. 
Moreover there exists an overhead for writing back the results to memory for every completed block of the result matrix.
This is the right side of the equation.
Since it just scales quadratic instead of the cubic scaling of the calculation for calculating an upper bound of the compute performance only the left side of the equation has to be considered.


#### Example for Bittware 520N Board

For the Bittware 520N board equipped with Stratix 10 FPGA the following constants are given:

- s_page = 1KB
- t_RCD and t_RP = 10.8ns
- s_block = 256 * 4B
- s_reg = 8 * 4B
- s_bus = 64B

So we can calculate the latencies per block:

![t_{mem}=\frac{4096}{f}+5.53µs](https://latex.codecogs.com/gif.latex?t_{mem}=\frac{4096}{f}+5.53\mu&space;s) 


![t_{calc}=\frac{32768}{f}](https://latex.codecogs.com/gif.latex?t_{calc}=\frac{32768}{f}) 

For a kernel frequency of 300MHz this would end up to t_mem=19.2µs<t_calc=109.2µs.
So the calculation will be compute bound.

For a matrix of size 4096 this leads to the total runtime of:

![t_{total}=4096*109.2\mu&space;s+256*19.2\mu&space;s=452ms](https://latex.codecogs.com/gif.latex?t_{total}=4096*109.2\mu&space;s+256*19.2\mu&space;s=452ms) 

This would resemble 304 GFLOPS.
The measured performance will be slightly lower because of the pipeline overhead.
The described model is implemented in a [Python script](performance/performance_model_cannon.py).

## Comparison to Execution Measurements

We synthesized the kernel for the Bittware 520N board with the following configuration:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
 `DATA_TYPE`     | float       | Data type used for calculation       |
`DEFAULT_DEVICE` | -1          | Index of the default device (-1 = ask) |
`DEFAULT_PLATFORM`| -1          | Index of the default platform (-1 = ask) |
`DEFAULT_REPETITIONS`| 10          | Number of times the kernel will be executed |
`KERNEL_NAME`| gemm | Name of the kernel (only needed for own implementations) |
`FPGA_BOARD_NAME`| p520_hpc_sg280l | Name of the target board |
`BLOCK_SIZE`    | 256          | Block size used by the kernel to transpose the matrix |
`GEMM_SIZE`    | 8             | Block size of the fully unrolled cannon block if cannon kernel is used |
`GLOBAL_MEM_UNROLL`| 16        | Unrolling factor for the global memory access |
`AOC_FLAGS`| `-fpc -fp-relaxed -no-interleaving=default` | Additional AOC compiler flags that are used for kernel compilation |

The versions of the used tools where:

Tool             | Version |
---------------- |---------|
Intel OpenCL SDK | 19.4.0  |
BSP              | 19.2.0  |
GCC              | 8.3.0   |

Output of the execution:

    -------------------------------------------------------------
    General setup:
    C++ high resolution clock is used.
    The clock precision seems to be 1.00000e+01ns
    -------------------------------------------------------------
    Selected Platform: Intel(R) FPGA SDK for OpenCL(TM)
    Multiple devices have been found. Select the platform by typing a number:
    0) p520_hpc_sg280l : BittWare Stratix 10 OpenCL platform (aclbitt_s10_pcie0)
    1) p520_hpc_sg280l : BittWare Stratix 10 OpenCL platform (aclbitt_s10_pcie1)
    0
    Enter device id [0-1]:-------------------------------------------------------------
    Selection summary:
    Platform Name: Intel(R) FPGA SDK for OpenCL(TM)
    Device Name:   p520_hpc_sg280l : BittWare Stratix 10 OpenCL platform (aclbitt_s10_pcie0)
    -------------------------------------------------------------
    -------------------------------------------------------------
    FPGA Setup:gemm_cannon.aocx
    MMD INFO : Disabling SmartVID (fix) polling
    MMD INFO : Enabling SmartVID (fix) polling
    Prepared FPGA successfully for global Execution!
    -------------------------------------------------------------
    Summary:
    Kernel Repetitions:  1
    Total matrix size:   8192
    Memory Interleaving: 0
    Kernel file:         gemm_cannon.aocx
    Device:              p520_hpc_sg280l : BittWare Stratix 10 OpenCL platform (aclbitt_s10_pcie0)
    -------------------------------------------------------------
    Start benchmark using the given configuration.
    -------------------------------------------------------------
             best         mean       GFLOPS
      3.38713e+00  3.38713e+00  3.24614e+02

The achieved kernel frequency is 320.8MHz and the measured compute performance is 324.6 GFLOPS.
The model gives 326.7 GFLOPS for the given frequency and matrix size, so the residual error of the model is 0.64%.
 