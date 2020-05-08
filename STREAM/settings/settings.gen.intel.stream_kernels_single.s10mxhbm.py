
global_memory_names = [ "HBM%d" % i for i in range(4) ]

def generate_attributes(num_replications):
    return [ "__attribute__((buffer_location(\"%s\")))" 
            % (global_memory_names[i % len(global_memory_names)])
            for i in range(num_replications)]