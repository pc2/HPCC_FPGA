# Plot and Data Scripts

The Python scripts in this folder can be used to parse the output of the STREAM benchmark
to the CSV format and plot the data.

### Dependencies

To run the scripts, Python 3 has to be installed.
Also the following Python packages need to be installed:

- Numpy
- Matplotlib
- Pandas

They can be installed using the `requirements.txt` file in this folder with:

    pip install -r requirements.txt

Both scripts can retrieve the data from files, folder or over the standard input.
For further information run the scripts with the help option.

Example use:

    ./parse_data.py -i input.txt | create_plots.py
   
This will load the output of the benchmark execution from a file `input.txt` and parse it
to CSV. The CSV format will be piped to the next script that will use it to generate the plots.

