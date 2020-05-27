
#include "hpcc_benchmark.hpp"

std::ostream& hpcc_base::operator<<(std::ostream& os, hpcc_base::BaseSettings const& printedBaseSettings){
        return (os  << "Repetitions:         " << printedBaseSettings.numRepetitions
              << std::endl
              << "Kernel File:         " << printedBaseSettings.kernelFileName
              << std::endl);
}
