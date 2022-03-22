import argparse
import sys
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

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.file:
        print('no input file given')
        exit(1)
    if not args.output_file: 
        print('no output file given')
        exit(1)
    for p in args.params:
        print("Parse statement: %s" % p)
        exec(p, globals())

    template = env.get_template(args.file)

    try:
        template.globals.update({"generate_attributes": generate_attributes})
    except:
        generate_attributes = lambda r : ["" for i in range(r)]
        template.globals.update({"generate_attributes": generate_attributes})

    if num_replications is None:
        num_replications = 1 

    with open(args.output_file, 'w') as f:
        f.write(template.render(num_replications=num_replications))