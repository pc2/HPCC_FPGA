import argparse
import sys
import logging
from jinja2 import Environment, PackageLoader, BaseLoader, TemplateNotFound, select_autoescape
from os.path import join, exists, getmtime

parser = argparse.ArgumentParser(description='Preprocessor for code replication and advanced code modification.')
parser.add_argument('file', metavar='CODE_FILE', type=str,
                   help='Path to the file that is used as input')
parser.add_argument("-o", dest="output_file", default=None, help="Path to the output file. If not given, output will printed to stdout.")
parser.add_argument("-p", dest="params", default=[], action="append", help="Python statement that is parsed before modifying the files. Can be used to define global variables.")

# create a simple loader to load templates from the file system
class SimpleLoader(BaseLoader):
    def __init__(self, path):
        self.path = path

    def get_source(self, environment, template):
        path = join(self.path, template)
        if not exists(path):
            raise TemplateNotFound(template)
        mtime = getmtime(path)
        with open(path) as f:
            source = f.read()
        return source, path, lambda: mtime == getmtime(path)

env = Environment(
    loader=SimpleLoader("./"),
    autoescape=select_autoescape()
)

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

if __name__ == '__main__':
    args = parser.parse_args()

    if args.output_file:
        log_file_name = args.output_file + ".log"
    else:
        log_file_name = "generator.log"
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.DEBUG)

    if not args.file:
        logging.debug('no input file given')
        exit(1)
    if not args.output_file: 
        logging.debug('no output file given')
        exit(1)
    for p in args.params:
        logging.debug("Parse statement: %s" % p)
        exec(p, globals())

    template = env.get_template(args.file)

    try:
        template.globals.update({"generate_attributes": generate_attributes})
    except:
        generate_attributes = lambda r : ["" for i in range(r)]
        template.globals.update({"generate_attributes": generate_attributes})

    try:
        template.globals.update({"generate_map_attributes": generate_map_attributes})
    except:
        generate_map_attributes = lambda r : [{"a": "", "b": "", "c": "", "out": ""} for i in range(r)]
        template.globals.update({"generate_map_attributes": generate_map_attributes})

    try:
        template.globals.update({"generate_bi_map_attributes": generate_bi_map_attributes})
    except:
        generate_bi_map_attributes = lambda r : [{"in": "", "out": ""} for i in range(r)]
        template.globals.update({"generate_bi_map_attributes": generate_bi_map_attributes})

    if num_replications is None:
        num_replications = 1 

    if num_total_replications is None:
        num_total_replications = 1

    with open(args.output_file, 'w') as f:
        f.write(template.render(num_replications=num_replications, num_total_replications=num_total_replications))