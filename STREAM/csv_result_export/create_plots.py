#!/usr/bin/env python3
from io import StringIO
import pandas as pd
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

beautify_map = {
    "Copy": "Copy",
    "Scale": "Scale",
    "Add": "Add",
    "Triad": "Triad",
    "PCIRead": "PCIe read",
    "PCIWrite": "PCIe write",
    "fmax": "max. Frequency"
}


def plot_polar(data_content):
    df = pd.read_csv(StringIO(data_content),index_col=0).T
    cols = [r for r in df.columns]
    used_bw_rows = [r for r in df.index if "PCI" not in r and "rate" in r and "total_rate" not in r]
    rotation = (np.pi / 4)

    # create polar plots
    fig1 = plt.figure(figsize=(6,3))
    ax = fig1.add_subplot(121,projection='polar')
    ax.set_xticklabels([beautify_map[r.split('_')[0]] for r in used_bw_rows])
    ax.set_xticks(np.array(range(len(used_bw_rows))) * (2 * np.pi) / len(used_bw_rows) + rotation)
    for col in cols:
        xv = np.array(list(range(len(used_bw_rows))) + [0]) * (2 * np.pi) / len(used_bw_rows) + rotation
        yv = list(df[col].T[used_bw_rows].values / 1000) + [df[col].T[used_bw_rows[0]] / 1000]
        col_label = col
        ax.plot(xv, yv, label=col_label, zorder=10)
    ax.grid(linestyle="dotted")
    ax.set_ylabel("GB/s")
    box = ax.get_position()
    ax.set_position([box.x0*0.4, box.y0*0.2, box.width * 1.3, box.height*1.3])
    ax.legend(loc='upper right', bbox_to_anchor=(2, 1),
              ncol=1, fancybox=True)
    fig1.savefig("kernel_bw_polar.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Plots from CSV files")
    parser.add_argument('-i', dest="input_path",
                        help="Path to CSV file that should be used to create plots",
                        default="-")
    args = parser.parse_args()
    # Select, if stdin and stdout or files should be used
    if args.input_path == "-":
        file_content = ""
        isopen = True
        while isopen:
            t = sys.stdin.read()
            if t == "":
                isopen = False
            file_content += t
        plot_polar(file_content)
    else:
        plot_polar(args.input_path)
