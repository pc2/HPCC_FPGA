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
import itertools
import sys
import logging
import re


comment_symbol = "//"
ml_comment_symbol_start = "/*"
ml_comment_symbol_end = "*/"
pycodegen_cmd = "PY_CODE_GEN"
pragma_cmd = comment_symbol +"\\s*"+ pycodegen_cmd

parser = argparse.ArgumentParser(description='Preprocessor for code replication and advanced code modification.')
parser.add_argument('file', metavar='CODE_FILE', type=str,
                   help='Path to the file that is used as input')
parser.add_argument("-o", dest="output_file", default=None, help="Path to the output file. If not given, output will printed to stdout.")
parser.add_argument("--comment", dest="comment_symbol", default=comment_symbol, help="Symbols that are used to comment out lines in the target language. Default='%s'" % comment_symbol)
parser.add_argument("--comment-ml-start", dest="comment_symbol_ml_start", default=ml_comment_symbol_start, help="Symbols that are used to start a multi line comment in the target language. Default='%s'" % ml_comment_symbol_start)
parser.add_argument("--comment-ml-end", dest="comment_symbol_ml_end", default=ml_comment_symbol_end, help="Symbols that are used to end a multi line comment in the target language. Default='%s'" % ml_comment_symbol_end)
parser.add_argument("-p", dest="params", default=[], action="append", help="Python statement that is parsed before modifying the files. Can be used to define global variables.")

CODE = ""

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


def replace(code_block=None, local_variables=None):
    """
    Evaluate or execute inline code and replace the code with the result.

    @param code_block The input code block that will be parsed and modified
    @param local_variables A dictionary containing local variables that should also be considered (like locals())

    @return the modified code
    """
    global CODE
    if not code_block:
        code_block = CODE
    if local_variables is not None:
        variables = {**globals(), **local_variables}
    else:
        variables = globals()
    matches = itertools.chain(re.finditer("%s\\s*%s\\s+(?P<code>(.|\n)+?)%s" % (ml_comment_symbol_start, pycodegen_cmd, ml_comment_symbol_end), code_block, flags=0),
                                re.finditer("%s\\s+((?!block_start)|(?!block_end))\\s+(?P<code>(.)+?)\n" % (pragma_cmd), code_block, flags=0))
    for res_ml in matches:
        logging.debug("Found inline code!")
        res_ml_code = res_ml.group(0)
        try:
            code_block = code_block.replace(res_ml_code, str(eval(res_ml.groupdict()["code"], variables)))
            continue
        except Exception as e:
            logging.debug("Failed to evaluate inline code")
        try:
            logging.debug("Try execution in global space")
            exec(res_ml.groupdict()["code"], globals())
            code_block = code_block.replace(res_ml_code, "")
        except Exception as e:
            logging.warning("Could not execute inline code:\n\tCommand: '''\n%s\n'''\n\tError: %s" % (res_ml.groupdict()["code"], e))
    return code_block


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
    return mod_code
    #logging.debug("Start parsing of modified sub-block")
    #parse_string(mod_code, out)
    #logging.debug("Finished parsing of modified sub-block")


def parse_string(code_string, out):
    try:
        code_string = replace(code_string)
        for res in re.finditer("%s\\s*block_start\\s+(?P<cmd>.*)\n(?P<code>(.|\n)+?)%s\\s*block_end\\s*\n" % (pragma_cmd, pragma_cmd), code_string, flags=0):
            logging.debug("Found block match!")
            d = res.groupdict()
            code_block = d["code"]
            logging.debug("Modify the block!")
            code_block = modify_block(code_block, d["cmd"], out)
            code_string = code_string.replace(res.group(0), code_block)
        logging.debug("Parsing complete. Write result to file.")
        output.write(code_string)
    except Exception as e:
        logging.error("Block: %s \n %s" % (code_string, e))
        logging.error("Global variables: %s" % globals())
        logging.error("Local variables: %s" % locals())
        print( "Error while parsing code block: %s \n %s" % (e),file=sys.stderr)


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
    if args.output_file:
        log_file_name = args.output_file + ".log"
    else:
        log_file_name = "generator.log"
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.DEBUG)
    output = sys.stdout
    for p in args.params:
        logging.debug("Parse statement: %s" % p)
        exec(p, globals())
    if args.output_file:
        logging.debug("Use output file: %s" % args.output_file)
        output = open(args.output_file, 'w')
    comment_symbol = re.escape(args.comment_symbol)
    ml_comment_symbol_start = re.escape(args.comment_symbol_ml_start)
    ml_comment_symbol_end = re.escape(args.comment_symbol_ml_end)
    pragma_cmd = comment_symbol +"\\s*"+ pycodegen_cmd
    logging.debug("Use pragma command: %s", pragma_cmd)
    logging.debug("Start parsing file: %s" % args.file)
    parse_file(args.file, output)
