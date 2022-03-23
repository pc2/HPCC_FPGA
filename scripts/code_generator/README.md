# Code Generation and Replication Tool

This is a small and highly extendable Python script for Code generation.
The main application area is the generation of OpenCL code, but the generator works independently of the used programming language.
It can be seen as an extension of the usually used preprocessors to adapt the code before compilation.
With this code it is also possible to replicate code sections and do more complex modifications while keeping the code readable.
This is done using inline scripting in code comments.
A generator code line always starts with `PY_CODE_GEN`.

## Execution

The script needs Python3 to run.
It will be used by the CMake build system to generate source code and settings for some of the benchmarks.
A short summary of the usage of the script that can also be printed by running `./generator.py -h`:

    usage: generator.py [-h] [-o OUTPUT_FILE] [--comment COMMENT_SYMBOL]
                        [--comment-ml-start COMMENT_SYMBOL_ML_START]
                        [--comment-ml-end COMMENT_SYMBOL_ML_END] [-p PARAMS]
                        CODE_FILE

    Preprocessor for code replication and advanced code modification.

    positional arguments:
    CODE_FILE             Path to the file that is used as input

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTPUT_FILE        Path to the output file. If not given, output will
                            printed to stdout.
    --comment COMMENT_SYMBOL
                            Symbols that are used to comment out lines in the
                            target language. Default='//'
    --comment-ml-start COMMENT_SYMBOL_ML_START
                            Symbols that are used to start a multi line comment in
                            the target language. Default='/*'
    --comment-ml-end COMMENT_SYMBOL_ML_END
                            Symbols that are used to end a multi line comment in
                            the target language. Default='*/'
    -p PARAMS             Python statement that is parsed before modifying the
                            files. Can be used to define global variables.



## Code Examples

The generator takes arbitrary code files as input and only applies changes when specific comment patterns are found.
The code insertions have the following syntax:

    // PY_CODE_GEN [block_start STATEMENT|block_end|STATEMENT]

it is also possible to write multiple lines of code:

    /* PY_CODE_GEN 
    STATEMENT1
    STATEMENT2
    ...
    */

Where `STATEMENT`is an arbitrary python statement.
The input file will be parsed from the beginning to the end and generation statements will be executed immediately.
Example for the definition of a global variable:

    PY_CODE_GEN replicate=4

This variable can then be used within the next pragmas to further modify the code.
E.g. the defined variable can be used to modifiy a code block:

    // PY_CODE_GEN block_start CODE.replace("$R", str(replicate))
    int i = $R;
    printf("i should be $R");
    // PY_CODE_GEN block_end 

`CODE` is a global variable containing the code within the recent block. It can be modified like every other Python string.
In most cases it is recommended to use the build-in function `replace()` for replacing variables, but it might be used for more advanced code modifications.
The result of the given Python statement will then be printed in the modified file.

This is functionality, which would also be possible using the standard preprocessor.
A case, where this script becomes handy is code replication.
This can easily be doe using list comprehension.
As an example the dynamic construction of a switch statement:

    switch(i) {
        // PY_CODE_GEN block_start [replace(local_variables=locals()) for i in range(replicate)]
        case {{ i }}: return /*PY_CODE_GEN i+1*/; break;
        // PY_CODE_GEN block_end 
    }

would result in:

    switch(i) {
        case 0: return 1; break;
        case 1: return 2; break;
        case 2: return 3; break;
        case 3: return 4; break;
    }

Note, that the variables that have to be replaced are written in inline comments `{{ i }}`.
The given statement will be evaluated and the comment will be replaced by the result.
Thus, it is also possible to call functions or do arithmetic.

## Built-In Functions

The generator can easily be extended by including additional file with the `use_file(FILENAME)` command.

    PY_CODE_GEN use_file(helpers.py)

This will read the file and make all functions and global variables available within following blocks.

`replace()` makes it easier to replace global variables within the code:

    // PY_CODE_GEN block_start replace(local_variables={"test": 2})
    int var = /*PY_CODE_GEN test*/
    // PY_CODE_GEN block_end

will generate the code `int var = 2`.

It is easily possible to add other helper functions and extend the functionality of the generator using the `use_file` method
or by declaring functions in multi line comments.
