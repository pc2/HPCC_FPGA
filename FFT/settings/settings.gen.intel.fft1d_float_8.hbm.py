
global_memory_name = "HBM"

def generate_attributes(num_replications, num_global_memory_banks=32):
    """
    Generates the kernel attributes for the global memory. They specify in which 
    global memory the buffer is located. The buffers will be placed using a 
    round robin scheme using the available global memory banks and the number of
    replications that should be generated (e.g. if a global memory contains multiple banks)

    @param num_replications Number okernel replications
    @param num_global_memory_banks Number of global memory banks that should be used for generation

    @return Array of maps. Maps contain two keys: "in" and "out" for the attributes assigned to input and output 
            buffers in globa memory.
    """
    global_memory_names = [ "%s%d" % (global_memory_name, i) for i in range(num_global_memory_banks)]
    return [ { "in": "__attribute__((buffer_location(\"%s\")))" 
            % (global_memory_names[(2*i) % num_global_memory_banks]),
            "out": "__attribute__((buffer_location(\"%s\")))" 
            % (global_memory_names[(2*i + 1) % num_global_memory_banks])}
            for i in range(num_replications)]