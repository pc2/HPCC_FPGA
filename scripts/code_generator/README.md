# Code generation and Replication Tool

This is a small and highly extendable Python script for Code generation.
The main application area is the generation of OpenCL code, but the generator works independently of the used programming language.
It can be seen as an extension of the usually used preprocessors to adapt the code before compilation.
With this code it is also possible to replicate code sections and do more complex modifications while keeping the code readable.
This is done using inline scripting in commented out code.
A generator code line always starts with `PY_CODE_GEN`.

## Usage

The generator takes arbitrary code files as input and only applies changes when specific pragmas are found.
The pragmas have the following syntax:

    PY_CODE_GEN [block_start|block_end STATEMENT|STATEMENT]

Where `STATEMENT`is a python statement.
It is possible to execute simple statements, but also to define nested blocks, which can be changed or replicated.
Example for the definition of a global variable:

    PY_CODE_GEN replicate=4

This variable can then be used within the next pragmas to further modify the code.
E.g. the defined variable can be used to modifiy a code block:

    // PY_CODE_GEN block_start
    int i = $R;
    printf("i should be $R");
    // PY_CODE_GEN block_end CODE.replace("$R", str(replicate))

`CODE` is a global variable containing the code within the recent block. It can be modified like every other Python string.
The result of the given Python statement will then be printed in the modified file.

This is functionality, which would also be possible using the standard preprocessor.
A case, where this script becomes handy is code replication.
This can easily be doe using list comprehension.
As an example the dynamic construction of a switch statement:

    switch(i) {
        // PY_CODE_GEN block_start
        case $i$: return $i$; break;
        // PY_CODE_GEN block_end [replace(replace_dict=locals()) for i in range(replicate)]
    }

would result in:

    switch(i) {
        case 0: return 0; break;
        case 1: return 1; break;
        case 2: return 2; break;
        case 3: return 3; break;
    }

## Built-In Functions

The generator can easily be extended by including additional file with the `use_file(FILENAME)` command.

    PY_CODE_GEN use_file(helpers.py)

This will read the file and make all functions and global variables available within following blocks.

`if_cond(CONDITION, IF, ELSE)` can be used to easily return different values depending on the evaluation result of CONDITION:

    PY_CODE_GEN i=2
    PY_CODE_GEN block_start
    PY_CODE_GEN block_end if_cond(i < 1, "Hello", "World")

will return 'World'.

`replace()` makes it easier to replace global variables within the code:

    // PY_CODE_GEN i=2
    // PY_CODE_GEN block_start
    int var = $i$
    // PY_CODE_GEN block_end replace()

will generate the code `int var = 2`.

It is easily possible to add other helper functions and extend the functionality of the generator using the `use_file` method.


