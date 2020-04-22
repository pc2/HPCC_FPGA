# Raw Result Folder

This folder contains the single precision results for different FPGA boards and SDK versions.
The files contain the raw output from the benchmark execution and can be parsed with the 
script `parse_data.py` given in the parent directory.

The files are neamed after the following scheme:

    SDK_BSP_BOARD-INFO[_ni].txt
    
with the following meaning:

- **SDK**: The used version of the SDK with major and minor releases are separated by `-`. e.g. 18.1.1 would resemble `18-1-1`
- **BSP**: The version of the used board support packages with same version scheme than the SDK.
- **BOARD_INFO**: Information about the used FPGA board following the scheme `BOARD-NAME-(FPGA-NAME)[-SVM]`
                   giving the board name and FPGA name with `-` instead of spaces and an optional `-SVM` suffix, if the
                   OpenCL shared virtual memory functionality was used. e.g. `Bittware-385A-(Intel-Arria-10-GX1150)` 
- optional **ni**: Indicates if automatic memory interleaving by the compiler was used or not.
                    If `_ni` is appended then no interleaving by the compiler was used.
                    
All runs were executed with arrays over 100 million values.