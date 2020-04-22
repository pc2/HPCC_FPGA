# Performance Evaluation

## Performance Model

FFT1d kernel modelled here can be found in the Intel OpenCL Design Samples. 
The design follows the radix 2<sup>2</sup> FFT architecture, which consists of the following:

1. ld(N) radix-2 butterflies
2. trivial rotations at every even stage
3. non-trivial rotations at every odd stage. This is the twiddle factor multiplication computed after the stage's butterfly.
4. shuffling using shift registers

The FFT if fully pipelined and the FFT step is unrolled over all ld(N) stages.
Hence the performance is limited by the global memory to feed the pipeline with data.
We will focus on modeling the fetch kernel that is loading the data from memory.
The kernel pipeline can be expressed with the following equation:

![t_{mempipeline}=\frac{\frac{s_{FFT}}{s_{bus}}}{f}](https://latex.codecogs.com/gif.latex?t_{mempipeline}=\frac{\frac{s_{FFT}}{s_{bus}}}{f}) 

where ![s_{FFT}](https://latex.codecogs.com/gif.latex?s_{block}) is the number of bytes needed to load from global memory for the FFT i.e. 4096 * 8B for a 4096 FFT with single precision complex values.
![s_{bus}](https://latex.codecogs.com/gif.latex?s_{bus}) the bus width of the global memory in bytes.
![f](https://latex.codecogs.com/gif.latex?f) is the kernel frequency.
Moreover latency will be added to this operation for every DRAM page that has to be activated:

![t_{memoverhead}=\frac{s_{FFT}}{s_{page}}*\(t_{RCD}+t_{RP}\)](https://latex.codecogs.com/gif.latex?t_{memoverhead}=\frac{s_{FFT}}{s_{page}}*\(t_{RCD}+t_{RP}\))

where ![s_{page}](https://latex.codecogs.com/gif.latex?s_{page}) is the size of a DRAM page in bytes.
![t_{RCD}](https://latex.codecogs.com/gif.latex?t_{RCD}) and ![t_{RP}](https://latex.codecogs.com/gif.latex?t_{RP}) are the
row address to column address delay and the row precharge time.

So the total time for the memory accesses for a the calculation of a single FFT is:

![t_{mem}=t_{mempipeline}+t_{memoverhead}](https://latex.codecogs.com/gif.latex?t_{mem}=t_{mempipeline}+t_{memoverhead}) 

This model does not consider latencies of the calculation pipeline or of the memory but it holds for batched calculations where these latencies are hidden.
If memory interleaving is used, t_memoverhead is also hidden by the access to subsequent memory banks.

## Synthesis Results

The kernel was synthesized with the following configuration for the Bittware 520N board:

Name             | Default     | Description                          |
---------------- |-------------|--------------------------------------|
`DEFAULT_DEVICE` | -1          | Index of the default device (-1 = ask) |
`DEFAULT_PLATFORM`| -1          | Index of the default platform (-1 = ask) |
`FPGA_BOARD_NAME`| p520_hpc_sg280l | Name of the target board |
`DEFAULT_REPETITIONS`| 10          | Number of times the kernel will be executed |
`DEFAULT_ITERATIONS`| 5000          | Default number of iterations that is done with a single kernel execution|
`LOG_FFT_SIZE`   | 12          | Log2 of the FFT Size that has to be used i.e. 3 leads to a FFT Size of 2^3=8|
`AOC_FLAGS`| `-fpc -fp-relaxed` | Additional AOC compiler flags that are used for kernel compilation |

The used tool versions:

Tool             | Version |
---------------- |---------|
Intel OpenCL SDK | 19.4.0  |
BSP              | 19.2.0  |
GCC              | 8.3.0   |

The resulting output:

    -------------------------------------------------------------
    Implementation of the FFT benchmark proposed in the HPCC benchmark suite for FPGA.
    Version: 1.0
    -------------------------------------------------------------
    Summary:
    FFT Size:            4096
    Data Size:           5000 * FFT Size * sizeof(cl_float) = 8.19200e+07 Byte
    Repetitions:         10
    Kernel file:         fft1d_float_8.aocx
    Device:              p520_hpc_sg280l : BittWare Stratix 10 OpenCL platform (aclbitt_s10_pcie0)
    -------------------------------------------------------------
    Start benchmark using the given configuration.
    -------------------------------------------------------------
       res. error    mach. eps
      3.17324e-01  1.19209e-07
    
                           avg         best
       Time in s:  1.81336e-06  1.81170e-06
          GFLOPS:  1.35528e+02  1.35652e+02
          
So the FFT implementation achieved 135.7 GFLOPs with a kernel frequency of 297.5MHz.
The kernel uses memory interleaving so the model simplifies to:

![t_{mem}=\frac{\frac{4096}{8}}{297.5MHz}=1.72\mu&space;s](https://latex.codecogs.com/gif.latex?t_{mem}=\frac{\frac{4096}{8}}{297.5MHz}=1.72\mu&space;s) 

which shows an 5.2% difference to the measurement that resulted in 1.81µs.
The difference may be caused by the latencies of the global memory and the calculation pipeline.
Also the store of the FFT result may interfere with the load operations since they use the same memory banks.

