#!/usr/bin/env python3
#
# Copyright (c) 2019 Marius Meyer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##

import argparse
import sys
import logging


comment_symbol = "//"
pycodegen_cmd = "PY_CODE_GEN"
pragma_cmd = PRAGMA +" "+ pycodegen_cmd

parser = argparse.ArgumentParser(description='Preprocessor for code replication and advanced code modification.')
parser.add_argument('file', metavar='CODE_FILE', type=str,
                   help='Path to the file that is used as input')
parser.add_argument("-o", dest="output_file", default=None, help="Path to the output file. If not given, output will printed to stdout.")
parser.add_argument("--comment", dest="comment_symbol", default=comment_symbol, help="Symbols that are used to comment out lines in the target language. Default='%s'" % comment_symbol)
parser.add_argument("-p", dest="params", default=[], action="append", help="Python statement that is parsed before modifying the files. Can be used to define global variables.")

CODE = ""

def replace(input_code=None,replace_dict=None, frame="$"):
    """
    Replace variables in given code.
    Helper function to simplify code replacement
    """
    if input_code is None:
        input_code = CODE
    if replace_dict is None:
        replace_dict = {**locals(), **globals()}
    mod_code = input_code
    for k, v in replace_dict.items():
        mod_code = mod_code.replace(frame + str(k) + frame, str(v))
    return mod_code


def if_cond(condition, if_return, else_return):
    """
    Evaluate a condition to select a return value

    @param condition Statement that evaluates to a boolean
    @param if_return Value that is returned if condition is true
    @param else_return Value that is returned if condition is false

    @return if_return or else_return, depending on condition
    """
    if condition:
        return if_return
    else:
        return else_return


def use_file(file_name):
    """
    Read and execute a python script

    This function adds the content of a given file to the current Python environment.
    It can be used to define global variables or functions in a separate file, which can then be called
    within the pragmas to generate code

    @param file_name Path to the file relative to the current working directory

    @returns None
    """
    logging.debug("Try parsing file: %s" % file_name)
    try:
        with open(file_name) as f:
            exec(f.read(), globals())
    except Exception as e:
        logging.error("Error while parsing file %s" % file_name)
        logging.error(e)
        print("Error while parsing external file. See logs for more information.",file=sys.stderr)
        exit(1)


def modify_block(code_block, cmd_str, out):
    global CODE
    CODE  = code_block
    if cmd_str == "":
        cmd_str = "None"
    try:
        mod_code = eval(cmd_str, {**globals(), **locals()})
    except Exception as e:
        logging.error("Block: %s \n %s" % (code_block, e))
        logging.error("Global variables: %s" % globals())
        print( "Block: %s \n %s" % (code_block, e),file=sys.stderr)
        exit(1)
    if type(mod_code) is list:
        mod_code = "".join(mod_code)
    elif mod_code is None:
        mod_code = ""
    elif type(mod_code) is not str:
        logging.warning("%s is not a string. Automatic convert to string!" % mod_code)
        mod_code = str(mod_code)
    logging.debug("Start parsing of modified sub-block")
    parse_string(mod_code, out)
    logging.debug("Finished parsing of modified sub-block")


def parse_string(code_string, out):
    nested_level = 0
    current_block = ""
    for line_number, line in enumerate(code_string.split('\n')):
        line += "\n"
        sline = line.strip()
        if sline.startswith(pragma_cmd) and nested_level == 0:
            sline = sline.replace(pragma_cmd,"")
            if 'block_start' in sline:
                nested_level += 1
            elif 'block_end' in sline:
                logging.error("Block end before start! Invalid syntax!")
                print("Block end before start! Invalid syntax!", file=sys.stderr)
                exit(1)
            else:
                cmd_str = sline.strip()
                try:
                    exec(cmd_str, globals())
                except Exception as e:
                    logging.error("Block: %s \n %s" % (current_block, e))
                    logging.error("Global variables: %s" % globals())
                    print( "Block: %s \n %s" % (current_block, e),file=sys.stderr)
                    exit(1)
        elif sline.startswith(pragma_cmd):
            sline = sline.replace(pragma_cmd,"")
            if 'block_start' in sline:
                nested_level += 1
                current_block += line
            elif 'block_end' in sline:
                nested_level -= 1
                if nested_level == 0:
                    cmd_str = sline.replace('block_end', "").strip()
                    modify_block(current_block, cmd_str, out)
                    current_block = ""
                else:
                    current_block += line
            else:
                current_block += line
        elif nested_level > 0:
            current_block += line
        else:
            print(line, end='', file=out)


def parse_file(file_name, out):
    """
    Opens a single source code file and applies the changes to it.

    The function will output the modified source code into the given output stream.

    @param file_name The psth to the source code file relative to the current working directory
    @param out       Output stream that is used to output the modified source code
    """
    try:
        with open(file_name) as f:
            parse_string(f.read(), out)
    except Exception as e:
        logging.error("Error when opening and parsing file %s: %s" % (file_name, e))
        print("Error occurred when parsing file. See logs for more details.",file=sys.stderr)




if __name__=="__main__":
    args = parser.parse_args()
    log_file_name = args.file + ".genlog"
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.DEBUG)
    output = sys.stdout
    for p in args.params:
        logging.debug("Parse statement: %s" % p)
        exec(p, globals())
    if args.output_file:
        logging.debug("Use output file: %s" % args.output_file)
        output = open(args.output_file, 'w')
    pragma_cmd = args.pragma_cmd
    logging.debug("Use pragma command: %s", pragma_cmd)
    logging.debug("Start parsing file: %s" % args.file)
    parse_file(args.file, output)
