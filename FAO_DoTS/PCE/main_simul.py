import os, sys
sys.path.append(os.getenv("HOME"))

Your_route/

# === IMPORT MAIN FUNCTIONS ===
# These functions allow generating combinations of experimental parameters,
# applying filters, and automatically executing the defined experiments.

from src.exe_experiments import (
    casuistica_experimento,     # generates all possible parameter combinations
    filtrar_combinaciones,      # filters combinations according to a criterion (optional)
    ejecutar_experimentos       # executes experiments one by one
)

from src.auxiliar import num_qubits  # Computes the required number of qubits

# Function responsible for plotting the results obtained in each experiment
from src.grafica_csv import (
    graficar_coste             # generates and saves a plot from the experiment CSV file
)

# === 1. DEFINE EXPERIMENT PARAMETERS ===
# Each list represents a set of possible values for one experiment dimension.
# All possible combinations between the elements of these lists will be generated.
#   - Problem: type of quantum problem to solve (in this case, always "Factorization").
#   - Size: problem instance size or identifier.
#   - Optimiz: classical optimization algorithm (e.g., "COBYLA", "POWELL", "SLSQP"...).
#   - k: parameter controlling the compression or grouping of variables in the quantum circuit.

Problema = ["Factorization"]
Num = [821999]
Optimiz = ["DIFFERENTIALEVOLUTION"] #["PSO"] or ["QDPSO"]
k = [4]

# Global dictionary. If set to None, the default parameters defined in utilities.py are used.
optimizer_params = None

# === 2. GENERATE ALL EXPERIMENT COMBINATIONS ===
# 'casuistica_experimento' takes the lists defined above and returns
# a list containing all possible combinations of their elements.
# For example:
# [
#   ["MaxCut", 10, "COBYLA", 2],
#   ["MaxCut", 10, "POWELL", 2]
# ]
# Each combination represents an independent experiment.
combinaciones = casuistica_experimento(Problema, Num, Optimiz, k)

# === 3. RUN EXPERIMENTS AND PLOT RESULTS ===
# Each generated combination is executed independently.
# Internally, 'ejecutar_experimentos' is responsible for:
#   - Loading the corresponding graph/problem instance
#   - Building the quantum circuit with the appropriate parameters
#   - Running the optimization procedure (VQE)
#   - Saving the results into JSON and CSV files
#   - Returning the path to the CSV file containing the cost history
#
# Afterwards, 'graficar_resultados' uses this path to generate a plot
# showing the evolution of the cost function value across iterations.

# Maximum number of optimization iterations
maxiter = 10000

# Number of shots to perform
n_shots = 1  # Not used in this case because exact simulation is employed

for combo in combinaciones:

    print(f"\n🚀 Running experiment with parameters: {combo}")
    
    # The hyperparameter alpha is not used in the simulations
    alpha = 0.0
    beta = 0.0
    
    # Execute the experiment and obtain the CSV path containing the results
    ruta_csv, ruta_csv_iter = ejecutar_experimentos(exp_list=combo, optimizer_params=optimizer_params, alpha=alpha, beta=beta, maxiter=maxiter, n_shots=n_shots, nqpus = None, cunqa_str = "Simulation", family_name = None)
    
    # Generate and save the corresponding plot from the experiment CSV
    # graficar_coste(ruta_csv_iter)

    


