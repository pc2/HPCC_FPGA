import numpy as np
import matplotlib.pyplot as plt


def t_mem(block_size, unroll, f_max, page_size=256, t_rcd=10.8e-9, t_rp=10.8e-9):
    t_pipeline = (block_size ** 2) / unroll / f_max
    t_lat = block_size ** 2 / page_size * (t_rcd + t_rp)
    return t_pipeline + t_lat


def t_calc(block_size, reg_size, f_max):
    return (block_size / reg_size) ** 3 / f_max


def t_total(total_size, block_size, reg_size, unroll, f_max, page_size=256, t_rcd=10.8e-9, t_rp=10.8e-9):
    total_calc = t_calc(block_size, reg_size, f_max)
    total_overhead = t_mem(block_size, unroll, f_max, page_size, t_rcd, t_rp)
    return (total_size/block_size) ** 3 * (total_calc) + (total_size/block_size) ** 2 * total_overhead

def gflops(total_size, block_size, reg_size, unroll, f_max, page_size=256, t_rcd=10.8e-9, t_rp=10.8e-9):
    t = t_total(total_size, block_size, reg_size, unroll, f_max, page_size, t_rcd, t_rp)
    return (2* (total_size ** 3))/t * 1.0e-9


def plot_flops():
    block_sizes = np.array([2 ** i for i in range(8,11)])
    size = np.linspace(2**12,2**16,100)
    fig, ax = plt.subplots()
    ax.set_title("FLOPS for different Matrix Sizes with an 8x8x8 Multiplication in Registers")
    ax.set_ylabel("GFLOPS")
    ax.set_xlabel("Matrix Size")
    #ax.set_xscale('log')
    for i, b in enumerate(block_sizes):
        ax.plot(size, [gflops(f,b,8,16,320e6) for f in size])
    plt.tight_layout()
    fig.savefig("performance_model_cannon.png")


if __name__ == "__main__":
    plot_flops()
    print(t_total(4096*2,256,8,16,320.8e6))
    print(gflops(4096,256,8,16,300e6))

