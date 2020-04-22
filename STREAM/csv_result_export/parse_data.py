#!/usr/bin/env python
import pandas as pd
import os
from os import path
import re
import logging
import argparse
import sys

"""
This script will parse output of the STREAM benchmark for FPGA from stdout or from a file or multiple files
and return the data as a CSV

"""



"""
Some constants used for file parsing

"""

STREAM_REGEX = ("{name}:\\s+(?P<{field}_rate>\\d+\\.\\d+(e[\\+|-]d+)?)\\s+(?P<{field}_avg_time>\\d+\\.\\d+(e[\\+|-]d+)?)"
                     "\\s+(?P<{field}_min_time>\\d+\\.\\d+(e[\\+|-]d+)?)\\s+(?P<{field}_max_time>\\d+\\.\\d+(e[\\+|-]d+)?)")
FREQUENCY_REGEX = "fMax=(?P<fmax>\\d+\\.\\d+)"

STREAM_METRICS = ["Copy","Scale","Add","Triad","PCI Write","PCI Read"]

regex_list = [STREAM_REGEX.format(name=m, field=''.join(m.split(' '))) for m in STREAM_METRICS] + [FREQUENCY_REGEX]

"""
Start of the function definition

"""


def parse(path_to_source, recursive=False):
    """
    Main function that should be called when data has to be parsed.
    Will parse the file content of one or more files and return it as a DataFrame

    :param path_to_source: Path to the file or folder that should be parsed
    :param recursive: If True, will parse files in a folder recursively
    :return: A Dataframe containing the data parsed from all files in the folder (and subfolders)
    """
    df = pd.DataFrame()
    if path.isfile(path_to_source):
        s_result = parse_file(path_to_source)
        df = df.append(s_result)
    elif path.isdir(path_to_source):
        df = parse_dir(path_to_source, recursive)
    else:
        logging.error("Not a file or folder. Aborting!")
        sys.exit(1)
    return df


"""
Start of the definition of helper functions

"""


def parse_dir(folder_name, recursive=False):
    """
    Parse all files in the folder and subfolders, if wanted.

    :param folder_name: Path to the folder that should be used to find output files
    :param recursive: Will search recursively if true
    :return: A Dataframe containing the data parsed from all files in the folder (and subfolders)
    """
    df = pd.DataFrame()
    folder_content = [path.join(folder_name,f) for f in os.listdir(folder_name)]
    for fname in [f for f in folder_content if path.isfile(f) and f.endswith(".txt")]:
        s_result = parse_file(fname)
        df = df.append(s_result)
    if recursive:
        for fname in [f for f in folder_content if path.isdir(f)]:
            df_result = parse_dir(fname, recursive)
            df = df.append(df_result)
    return df


def parse_file(fname):
    """
    Open a file and pass the content to parse_content function

    :param fname: Path to the file
    :return: A Dataframe containing the data parsed from the file content
    """
    with open(fname) as f:
        return parse_content(f.read(), fname)


def parse_content(file_content, file_path):
    """
    Parse a string in to a DataFrame. Will use the given file path for logging

    :param file_content: Content of the file
    :param file_path: Path to the file
    :return: A Dataframe containing the data parsed from the file content
    """
    res = [list(re.finditer(r, file_content)) for r in regex_list]
    result_dicts = []
    for i,r in enumerate(res):
        if len(r) == 0:
            logging.warning("Some results could not be parsed with regex '%s' from %s" % (regex_list[i],file_path))
            res.remove(r)
    if any([len(res[0]) != len(r) for r in res]):
        logging.error("Different number of matches! Aborting! %s" % file_path)
        sys.exit(1)
    data_name = path.basename(file_path).split('.')[0]
    sdk_version = ".".join(data_name.split('_')[0].split("-"))
    no_interleaving = (data_name.split('_')[-1] != "ni")
    bsp_version = ".".join(data_name.split('_')[1].split("-"))
    device_name = " ".join(data_name.split('_')[2].split("-"))
    for i, x in enumerate(res[0]):
        result_dict = {"SDK": sdk_version, "no_interleaving": no_interleaving,
                       "Device" : device_name, "BSP": bsp_version}
        for y in res:
            d = y[i].groupdict()
            for k in d.keys():
                result_dict[k] = float(d[k])
        result_dicts.append(result_dict)

    if len(result_dicts) > 1:
        index_names = ["%s_%d" % (data_name, i) for i in range(len(result_dicts))]
    else:
        index_names = [data_name]
    return pd.DataFrame(result_dicts, index=index_names)


"""
Main code that will be executed when this script is directly called

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse plain text STREAM benchmark outputs to CSV")
    parser.add_argument('-i', dest="input_path",
                        help="Path to a text file containing the output of the STREAM benchmark or to a folder containing multiple output files",
                        default="-")
    parser.add_argument('-o', dest='output_file', help="Name of the output file", default="-")
    parser.add_argument('-r', dest='recursive', action='store_true', help="Recursively parse files in folders", default=False)
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
        df = parse_content(file_content, "stdout")
    else:
        df = parse(args.input_path, args.recursive)
    if args.output_file == "-":
        df.to_csv(sys.stdout, header=True)
    else:
        df.to_csv(args.output_file, header=True)
