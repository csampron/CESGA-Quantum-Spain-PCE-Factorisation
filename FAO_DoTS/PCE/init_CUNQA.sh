### ===================================================== ###
###   Module loading script for CUNQA HPC                ###
### ===================================================== ###
###
### This script is responsible for loading all modules
### required to run projects in the CUNQA environment.
### By executing:
###
###     source init_CUNQA.sh
###
### the necessary compilers, libraries, frameworks,
### and Python tools will be loaded for development
### and code execution on the cluster.
###

# Clear previously loaded modules
module purge

# Load main compiler, MPI, and library modules
module load qmio/hpc gcc/12.3.0 hpcx-ompi flexiblas/3.3.0 boost cmake/3.27.6 gcccore/12.3.0 nlohmann_json/3.11.3 ninja/1.9.0 pybind11/2.13.6-python-3.11.9 qiskit/1.2.4-python-3.11.9

# Load additional Python modules for graph processing and visualization
module load networkx/3.3-python-3.11.9
module load matplotlib/3.6.3-python-3.11.9