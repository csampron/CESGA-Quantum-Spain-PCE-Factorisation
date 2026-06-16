# CESGA-Quantum-Spain-PCE-Factorisation

<p align="center">
  <a>
    <img src="https://img.shields.io/badge/os-linux-blue" alt="Python version" height="18">
  </a>
  <a>
    <img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="Python version" height="18">
  </a>
</p>

<br>

<div align="center">
  <table>
    <tr>
      <td align="center" width="300">
        <a href="https://www.cesga.es/">
          <picture>
            <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/logo_cesga_blanco.png">
            <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/logo_cesga_negro.png">
            <img src="docs/source/_static/logo_cesga_negro.png" width="200" alt="CESGA logo">
          </picture>
        </a>
      </td>

      <td width="60"></td>

      <td align="center" width="300">
        <a href="https://quantumspain-project.es/">
          <picture>
            <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/QuantumSpain_logo_white.png">
            <source media="(prefers-color-scheme: light)" srcset="docs/source/_static/QuantumSpain_logo_color.png">
            <img src="docs/source/_static/QuantumSpain_logo_white.png" width="240" alt="QuantumSpain logo">
          </picture>
        </a>
      </td>
    </tr>
  </table>
</div>



In this repository you can find the code used to get the results shown in the paper Can PCE solve the factorisation problem via optimisation?

The repository is organized as followed:

- `FAO_DoTS`: Code used for the execution of the DoTS approach.
    - `Results_initial`: Directory with initial plots, results and the code used to obtain such plots and results
    - `Results`: Directory with final plots, results and the code used to obtain such plots and results
    - `PCE`: Code used to execute PSO, QDPSO and DE
        - `init_cunqa.sh` to download dependencies
        - `main_simul.sh` to execute the program
        - `src/` contains all the main modules to prepare, execute and analyse the experiments, it includes:
            - `auxiliar.py` → Auxiliar functions, including calculation of the number of qubits and coding of Hamiltonians.  
            - `exe_experiments.py` → Generation of experiment combinations and automated execution.
            - `grafica_csv.py` → Functions to generate figures based on the results shown on a CSV file. 
            - `utilities.py` → Classical optimisation (Powell, COBYLA, Differential Evolution, etc.) and callbacks.
            - `tensor_exp_value.py` → Tensor construction of the computational basis and qubits combinations.
            - `circuit_builder.py` → Class used to build and compile parametric circuits. 
            - `graphs/` → Instances of the MaxCut problem used on the benchmark.
    - `PCE_parallel`: Contains the same as PCE but can parallelize DE and SaCeSS using workers

- `FAO_ORG`: Code used for the execution of the Basic approach.
    - `Results_initial`: Directory with initial plots, results and the code used to obtain such plots and results
    - `Results`: Directory with final plots, results and the code used to obtain such plots and results
    - `PCE`: Code used to execute PSO, QDPSO and DE
        - `init_cunqa.sh` to download dependencies
        - `main_simul.sh` to execute the program
        - `src/` contains all the main modules to prepare, execute and analyse the experiments, it includes:
            - `auxiliar.py` → Auxiliar functions, including calculation of the number of qubits and coding of Hamiltonians.  
            - `exe_experiments.py` → Generation of experiment combinations and automated execution.
            - `grafica_csv.py` → Functions to generate figures based on the results shown on a CSV file. 
            - `utilities.py` → Classical optimisation (Powell, COBYLA, Differential Evolution, etc.) and callbacks.
            - `tensor_exp_value.py` → Tensor construction of the computational basis and qubits combinations.
            - `circuit_builder.py` → Class used to build and compile parametric circuits. 
            - `graphs/` → Instances of the MaxCut problem used on the benchmark.
    - `PCE_parallel`: Contains the same as PCE but can parallelize DE using workers

- `Resources`: Code used to obtain the computational resources graphics of the approaches
    - `auxiliar` → Directory with necessary functions to compute the circuit depth and the number of parameters
    - `depth_challengees_DoTS` → Functions that plots the depth of the circuits for the DoTS approach
    - `depth_challengees_ORG` → Functions that plots the depth of the circuits for the Basic approach
    - `params_challengees_DoTS` → Functions that plots the number of parameters of the circuits for the DoTS approach
    - `params_challengees_ORG` → Functions that plots the number of parameters of the circuits for the Basic approach
    - `qubits_challengees_DoTS` → Functions that plots the number of qubits of the circuits for the DoTS approach
    - `qubits_challengees_ORG` → Functions that plots the number of qubits of the circuits for the Basic approach

- `Numbers_test`: List of numbers that were used to factorise

### How to execute on HPC environment

In order to execute a problem of your liking you must change the `main_simul.py` file to your desired problem specifications
the following represents a given specification:

```console
Problema = ["Factorization"]                                     # Problem to solve
Num = [821999]                                                   # Number to factorise
Optimiz = ["DIFFERENTIALEVOLUTION"] #["PSO"] or ["QDPSO"]        # Classical algorithm optimizer, you can choose any optimizer you desire as long as their behavour is described on the `utilities.py` file
k = [2]                                                          # PCE compression parameter

```

After chosing the desired problem to solve:

1. Go to PCE directory on terminal `cd PCE`
2. Execute the comand `source init_cunqa.sh`
3. Execute the comand `sbatch main_simul.sh`, you can change the SBATCH specs inside this file



