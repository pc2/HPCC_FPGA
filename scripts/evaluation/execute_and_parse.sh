#!/bin/bash
#
# Executes a benchmark and pipes the result to the parser to get a CSV output
# Specify the benchmark commands over the command line options

SCRIPT_PATH=$( cd "$(dirname $0)"; pwd -P)

$@ | ${SCRIPT_PATH}/parse_raw_to_csv.py 

# Exit with the exit status of the parse command
exit $?
