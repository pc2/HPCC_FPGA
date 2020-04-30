
##
# This function will replicate the stream kernels to the desired amount given in NUM_REPLICATIONS
#
##
function(generate_kernel_replications)
    file(READ "${base_file}" file_content)
    file(WRITE "${source_f}" "")
    math(EXPR iterator "${ARGV0} - 1")
    foreach(number RANGE ${iterator})
        string(REGEX REPLACE "KERNEL_NUMBER" ${number} mod_file_content "${file_content}")
        file(APPEND "${source_f}" "${mod_file_content}")
        file(APPEND "${source_f}" "\n")
    endforeach()
endfunction()

generate_kernel_replications(${NUM_REPLICATIONS})
