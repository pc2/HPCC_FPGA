
# Allow reordeing of math operations to increase parallelism
config_compile -unsafe_math_optimizations

# Reduce number of memory ports to reduce resource uage for GMI
#config_interface -m_axi_auto_max_ports false