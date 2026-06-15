import os, sys
sys.path.append(os.getenv("HOME"))

from cunqa.qutils import qraise, qdrop

# === IMPORT MAIN FUNCTIONS ===
from src.exe_experiments import (
    casuistica_experimento,
    filtrar_combinaciones,
    ejecutar_experimentos
)

from src.auxiliar import num_qubits

from src.grafica_csv import (
    graficar_coste
)


def main():

    # === 1. DEFINE EXPERIMENT PARAMETERS ===
    Problema = ["Factorization"]
    Num = [10080582527]
    Optimiz = ["SACESS"]
    k = [4]

    # Optimizer parameters
    sacess_params = {
        "num_workers": 2,
        "max_walltime_s": 259200
    }

    optimizer_params = {
        "SACESS": sacess_params
    }

    # === 2. GENERATE ALL COMBINATIONS ===
    combinaciones = casuistica_experimento(Problema, Num, Optimiz, k)

    # === EXPERIMENT CONFIGURATION ===
    maxiter = 1
    n_shots = 1

    # === 3. RUN THE EXPERIMENTS ===
    for combo in combinaciones:

        print(f"\n🚀 Running experiment with parameters: {combo}")
        
        # Define hyperparameters
        alpha = 0.0
        beta = 0.0
        
        # Run experiment
        ruta_csv, ruta_csv_iter = ejecutar_experimentos(
            exp_list=combo,
            optimizer_params=optimizer_params,
            alpha=alpha,
            beta=beta,
            maxiter=maxiter,
            n_shots=n_shots,
            nqpus=None,
            cunqa_str="Simulation",
            family_name=None
        )
        
        # Plot results (optional)
        # graficar_coste(ruta_csv_iter)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()