#!/bin/bash
#SBATCH -J finplot
#SBATCH -o finplot_%j.out
#SBATCH -e finplot_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=30G

# === LOAD MODULE ===
module load qmio/hpc gcc/12.3.0 hpcx-ompi flexiblas/3.3.0 boost cmake/3.27.6 gcccore/12.3.0 nlohmann_json/3.11.3 ninja/1.9.0 pybind11/2.13.6-python-3.11.9 qiskit/1.2.4-python-3.11.9
module load qmio/hpc gcc/12.3.0 networkx/3.3-python-3.11.9
module load qmio/hpc gcccore/12.3.0 matplotlib/3.6.3-python-3.11.9


echo "Starting execution: $(hostname)"
echo "Date: $(date)"

srun python -u main_simul.py

echo "Execution ended: $(date)"



