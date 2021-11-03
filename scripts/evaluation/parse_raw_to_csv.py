#!/usr/bin/env python3

import argparse
import pandas as pd
import os
from os import path
import re
import io
import sys

# Regular expressions for the raw output of all 
fft_regex = "Version:\\s+(?P<version>.+)\n(.*\n)+Batch\\sSize\\s+(?P<batch_size>\d+)\n(.*\n)FFT\\sSize\\s+(?P<size>\d+)(.*\n)+Device\\s+(?P<device>.+)\n(.*\n)+\\s+res\.\\serror\\s+mach\.\\seps\n\\s+(?P<error>(\d|\.|\+|-|e)+)\\s+(?P<epsilon>(\d|\.|\+|-|e)+)(.*\n)+\\s+avg\\s+best\n\\s+Time\\s+in\\s+s:\\s+(?P<avg_time>(\d|\.|\+|-|e)+)\\s+(?P<best_time>(\d|\.|\+|-|e)+)\n\\s+GFLOPS:\\s+(?P<avg_flops>(\d|\.|\+|-|e)+)\\s+(?P<best_flops>(\d|\.|\+|-|e)+)"
gemm_regex = "Version:\\s+(?P<version>.+)\n(.*\n)+Matrix\\sSize\\s+(?P<size>\d+)(.*\n)+Device\\s+(?P<device>.+)\n(.*\n)+\\s+norm\.\\sresid\\s+resid\\s+machep\n\\s+(?P<error>(\d|\.|\+|-|e)+)\\s+(?P<resid>(\d|\.|\+|-|e)+)\\s+(?P<epsilon>(\d|\.|\+|-|e)+)(.*\n)+\\s+best\\s+mean\\s+GFLOPS\n\\s+(?P<best_time>(\d|\.|\+|-|e)+)\\s+(?P<avg_time>(\d|\.|\+|-|e)+)\\s+(?P<gflops>(\d|\.|\+|-|e)+)"
ra_regex = "Version:\\s+(?P<version>.+)\n(.*\n)+Array\\sSize\\s+(?P<size>(\d|\.|\+|-|e)+)(.*\n)+Kernel\\sReplications\\s+(?P<replications>\d+)(.*\n)+Device\\s+(?P<device>.+)\n(.*\n)+Error:\\s+(?P<error>(\d|\.|\+|-|e)+)(.*\n)+\\s+best\\s+mean\\s+GUOPS\n\\s+(?P<best_time>(\d|\.|\+|-|e)+)\\s+(?P<avg_time>(\d|\.|\+|-|e)+)\\s+(?P<gops>(\d|\.|\+|-|e)+)"
trans_regex = "Version:\\s+(?P<version>.+)\n(.*\n)+Matrix\\sSize\\s+(?P<size>\d+)(.*\n)+Device\\s+(?P<device>.+)\n(.*\n)+\\s*Maximum\\serror:\\s+(?P<error>(\d|\.|\+|-|e)+)(.*\n)+\\s+total\\s\\[s\\]\\s+transfer\\s\\[s\\]\\s+calc\\s\\[s\\]\\s+calc\\s+FLOPS\\s+Mem\\s+\\[B/s\\]\\s+PCIe\\s+\\[B/s\\]\n\\s*avg:\\s+(?P<avg_total_time>(\d|\.|\+|-|e)+)\\s+(?P<avg_transfer_time>(\d|\.|\+|-|e)+)\\s+(?P<avg_calc_time>(\d|\.|\+|-|e)+)\\s+(?P<avg_calc_flops>(\d|\.|\+|-|e)+)\\s+(?P<avg_mem_bw>(\d|\.|\+|-|e)+)\\s+(?P<avg_trans_bw>(\d|\.|\+|-|e|inf)+)\n\\s*best:\\s+(?P<best_total_time>(\d|\.|\+|-|e)+)\\s+(?P<best_transfer_time>(\d|\.|\+|-|e)+)\\s+(?P<best_calc_time>(\d|\.|\+|-|e)+)\\s+(?P<best_calc_flops>(\d|\.|\+|-|e)+)\\s+(?P<best_mem_bw>(\d|\.|\+|-|e)+)\\s+(?P<best_trans_bw>(\d|\.|\+|-|e|inf)+)"
stream_regex = "Version:\\s+(?P<version>.+)\n(.*\n)+Array\\sSize\\s+\\d+\\s+\\((?P<size>(\d|\.|\+|-|e)+)(.*\n)+Data\\sType\\s+(?P<data_type>.+)\n(.*\n)+Kernel\\sReplications\\s+(?P<replications>\d+)(.*\n)+Kernel\\sType\\s+(?P<type>.+)\n(.*\n)+Device\\s+(?P<device>.+)\n(.*\n)+\\s+Function\\s+Best\\sRate\\sMB/s\\s+Avg\\stime\\ss\\s+Min\\stime\\s+Max\\stime\n\\s+Add\\s+(?P<add_rate>(\d|\.|\+|-|e)+)\\s+(?P<add_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<add_min_time>(\d|\.|\+|-|e)+)\\s+(?P<add_max_time>(\d|\.|\+|-|e)+)\n\\s+Copy\\s+(?P<copy_rate>(\d|\.|\+|-|e)+)\\s+(?P<copy_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<copy_min_time>(\d|\.|\+|-|e)+)\\s+(?P<copy_max_time>(\d|\.|\+|-|e)+)\n\\s+PCI\\sread\\s+(?P<pcir_rate>(\d|\.|\+|-|e)+)\\s+(?P<pcir_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<pcir_min_time>(\d|\.|\+|-|e)+)\\s+(?P<pcir_max_time>(\d|\.|\+|-|e)+)\n\\s+PCI\\swrite\\s+(?P<pciw_rate>(\d|\.|\+|-|e)+)\\s+(?P<pciw_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<pciw_min_time>(\d|\.|\+|-|e)+)\\s+(?P<pciw_max_time>(\d|\.|\+|-|e)+)\n\\s+Scale\\s+(?P<scale_rate>(\d|\.|\+|-|e)+)\\s+(?P<scale_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<scale_min_time>(\d|\.|\+|-|e)+)\\s+(?P<scale_max_time>(\d|\.|\+|-|e)+)\n\\s+Triad\\s+(?P<triad_rate>(\d|\.|\+|-|e)+)\\s+(?P<triad_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<triad_min_time>(\d|\.|\+|-|e)+)\\s+(?P<triad_max_time>(\d|\.|\+|-|e)+)"
linpack_regex = "Version:\\s+(?P<version>.+)\n(.*\n)+Matrix\\sSize\\s+(?P<size>\d+)(.*\n)+Device\\s+(?P<device>.+)\n(.*\n)+\\s+norm\.\\sresid\\s+resid\\s+machep.+\n\\s+(?P<error>((\d|\.|\+|-|e)+|nan))\\s+(?P<resid>((\d|\.|\+|-|e)+|nan))\\s+(?P<epsilon>(\d|\.|\+|-|e)+)(.*\n)+\\s+Method\\s+\\s+best\\s+mean\\s+GFLOPS(\\s*\n)\\s+total\\s+(?P<total_best_time>(\d|\.|\+|-|e)+)\\s+(?P<total_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<total_gflops>(\d|\.|\+|-|e)+)(\\s*\n)\\s+GEFA\\s+(?P<lu_best_time>(\d|\.|\+|-|e)+)\\s+(?P<lu_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<lu_gflops>(\d|\.|\+|-|e)+)(\\s*\n)\\s+GESL\\s+(?P<sl_best_time>(\d|\.|\+|-|e)+)\\s+(?P<sl_avg_time>(\d|\.|\+|-|e)+)\\s+(?P<sl_gflops>(\d|\.|\+|-|e)+)"
   

def parse_network(file_content):
    '''
    The b_eff benchmark uses a special function since the input is just directly parsed as CS.

    file_content: Content of the file is parsed
    '''
    df = pd.DataFrame()
    regex = "(?P<data>\\s+MSize\\s+looplength\\s+transfer\\s+B/s\n(.+\n)+)"
    res = re.search(regex, file_content)
    if res is not None:
        d = res.groupdict()
        df = pd.read_csv(io.StringIO(d["data"]), sep="\\s+")
    else:
        return None
    return df


def parse_by_regex(file_content, regex, bm_name):
    '''
    Parsing function using a REGEX.

    file_content: Content of the file is parsed
    regex: The regular expression that is used to parse the text
    bm_name: Name of the benchmark. Will be used as index in the data frame
    '''
    df = pd.DataFrame()
    res = re.search(regex, file_content)
    if res is not None:
        d = res.groupdict()
        df_tmp = pd.DataFrame(d, index=[bm_name])
        df = df.append(df_tmp)
    else:
        return None
    return df

# The parsing functions for each benchmark preconfigured in a map
parse_map = {
    "b_eff": parse_network,
    "FFT": lambda f: parse_by_regex(f, fft_regex, "FFT"),
    "GEMM": lambda f: parse_by_regex(f, gemm_regex, "GEMM"),
    "LINPACK": lambda f: parse_by_regex(f, linpack_regex, "LINPACK"),
    "PTRANS": lambda f: parse_by_regex(f, trans_regex, "PTRANS"),
    "RandomAccess": lambda f: parse_by_regex(f, ra_regex, "RandomAccess"),
    "STREAM": lambda f: parse_by_regex(f, stream_regex, "STREAM")
}

def parse_single_file(file_name, used_parse_functions):
    # Read file content from stdin or a given file
    if file_name == "-":
        file_content = ""
        isopen = True
        while isopen:
            t = sys.stdin.read()
            if t == "":
                isopen = False
            file_content += t
    else:
        with open(file_name) as f:
            file_content = f.read()

    # Try to parse the file content
    for b in used_parse_functions:
        df = b(file_content)
        if not df is None:
            break
    if df is None:
        print("File content could not be parsed: %s" % file_name, file=sys.stderr)
    df['filename'] = file_name
    return df


def parse_file_or_folder(file_name, used_parse_functions):
    df = pd.DataFrame()
    if os.path.isdir(file_name):
        files_in_dir = os.listdir(file_name)
        for f in files_in_dir:
            df = df.append(parse_file_or_folder(f, used_parse_functions))
    else:
        tmp = parse_single_file(file_name, used_parse_functions)
        if not tmp is None:
            df = df.append(tmp)
    return df
        
def parse_raw_inputs(input_paths, recursive=True, parse_functions=parse_map):

    if type(input_paths) is not list:
        input_paths = list(input_paths)

    df = pd.DataFrame()
    for ifile in input_paths:
        if recursive:
            df = df.append(parse_file_or_folder(ifile, parse_functions))
        elif not os.path.isdir(ifile):
            df = df.append(parse_single_file(ifile, parse_functions))
        else:
            print("Directory was specified, but no recursive execution", file=sys.stderr)
    return df

def parse_script_called_directly():
    # Define input parameters
    parser = argparse.ArgumentParser(description="Parse plain text outputs of HPCC benchmarks to CSV")
    parser.add_argument('-i', dest="input_paths", nargs='+',
                        help="Path to a text file containing the output of an HPCC benchmark. If not given, stdin is used.",
                        default="-")
    parser.add_argument('-r', dest='recursive', action='store_const',
                    const=True, default=False, help="Recursively parse files in a folder")
    parser.add_argument('-b', dest='benchmark', help="Restrict parsing just to the named benchmark. Valid names are: %s" % list(parse_map.keys()), default="-")
    parser.add_argument('-o', dest='output_file', help="Name of the output file. If not given stout is used.", default="-")
    args = parser.parse_args()

    # If a benchmark restriction is given just use its parsing function
    used_parse_functions = parse_map.values()
    if args.benchmark in parse_map.keys():
        used_parse_functions = [parse_map[args.benchmark]]

    df = parse_raw_inputs(args.input_paths, args.recursive, used_parse_functions)

    if df is None:
        print("No files could be parsed", file=sys.stderr)
        exit(1)

    # Write the resulting CSV data to stdout or a file
    if args.output_file == "-":
        df.to_csv(sys.stdout, header=True)
    else:
        df.to_csv(args.output_file, header=True)


if __name__ == "__main__":
    parse_script_called_directly()
