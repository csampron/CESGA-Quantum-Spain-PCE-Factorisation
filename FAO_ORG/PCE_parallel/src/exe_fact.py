### ========================================================= ###
### Módulo: exe_fact
### ========================================================= ###
###
### Funciones para:
### - Ejecutar experimentos de MaxCut con ansatz cuántico variacional (VQE)
### - Compilar circuitos y medir en bases X, Y, Z
### - Optimización clásica y refinamiento local
### - Guardar resultados parciales y finales en CSV/JSON
###
### ========================================================= ###

def append_result_to_json(file_path, new_result):
    """
    Añade un diccionario 'new_result' a un archivo JSON con formato:
    {"resultados": [ ... ]}, usando bloqueo exclusivo (flock).
    """
    import json
    import os
    import fcntl

    # Asegura que el directorio existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Abrimos el archivo en modo lectura/escritura o lo creamos si no existe
    with open(file_path, "a+", encoding="utf-8") as f:
        f.seek(0)  # Ir al inicio
        # Bloqueo exclusivo: otro proceso esperará hasta que se libere
        fcntl.flock(f, fcntl.LOCK_EX)

        try:
            # Si el archivo está vacío, inicializamos la estructura
            try:
                f.seek(0)
                data = json.load(f)
                if not isinstance(data, dict) or "resultados" not in data:
                    data = {"resultados": []}
            except json.JSONDecodeError:
                data = {"resultados": []}

            # Añadir nuevo resultado
            data["resultados"].append(new_result)

            # Reescribir el archivo completo
            f.seek(0)
            f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=2)

        finally:
            # Liberar el bloqueo
            fcntl.flock(f, fcntl.LOCK_UN)

def ejecutar_fact(
    N_int,
    optimizer,
    optimizer_params, 
    k,
    alpha,
    beta,
    maxiter,
    n_shots,
    nqpus,
    cunqa_str_arg,
    family_name
):
    """
    Ejecuta un experimento de MaxCut utilizando un circuito cuántico variacional (VQE).
    Construye el circuito, ejecuta la optimización, guarda resultados parciales en JSON/CSV
    y devuelve los resultados de la ejecución.
    """

    # === IMPORTS INTERNOS ===
    import math
    import time
    import numpy as np
    
    from qiskit import transpile, QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit import ClassicalRegister
   
    import json
    import os

    # Módulos del proyecto
    from src.loss_functions import loss_func_estimator
    from src.auxiliar import bits_required, worst_case_bits_for_primes, num_qubits, alt_extract_primes
    from src.utilities import run_vqe_optimization
    from src.circuit_builder import Circuit
    from src.tensor_exp_value import build_sign_tensor
    
    # === 1. Cargar el grafo del problema ===
    bits_req = bits_required(N_int)
    print(f"N: {N_int}, Num_N_bits: {bits_req}")

    # === 2. Definir número de qubits y capas del circuito ===
    
    num_p_bits, num_q_bits = worst_case_bits_for_primes(bits_req)
    m = num_p_bits + num_q_bits 

    print(f"Numero de bits de p + q: {m}")
  

    qubits = num_qubits(m, k)                   # número de qubits en función de vértices y k
    layers = m**(1 - (1 / k))                 # fórmula para el número de capas

    layers = int(np.ceil(1.5 * (qubits ** (np.floor(k/2))))) 

    num_layers = math.ceil(layers)                    # redondeo al entero superior

    # === 3. Construir codificaciones de Pauli ===
    # Se dividen los nodos en tres subconjuntos (X, Y, Z)
    list_size = m // 3
    
    # === 4. Crear y compilar el circuito cuántico ===
    sim = AerSimulator()   # simulador de Qiskit Aer
    

    # Constructor del circuito con los parámetros del experimento

    qc_builder = Circuit(
        size=qubits,
        p=num_layers,
        entanglement='Taylor_efficient',
        rotation='Taylor_efficient',
        connectivity='brickwork_single_rotating',
        mode='custom'
    )

    
    qc_builder.compile_circuit()          # construir el circuito
    qc = qc_builder.get_circuit()         # obtener el circuito final

    if cunqa_str_arg != "Simulation":

        # --- Crear copias del circuito original para medir en cada base ---
        qc_z = qc.copy()  # Para términos Z
        qc_x = qc.copy()  # Para términos X
        qc_y = qc.copy()  # Para términos Y

        sim = AerSimulator(seed_simulator=33)   # simulador de Qiskit Aer
        # --- Añadir rotaciones de base y medidas solo si no es Simulation ---
        
        # Z: medir directamente
        cr_z = ClassicalRegister(qubits)
        qc_z.add_register(cr_z)
        qc_z.measure(range(qubits), range(qubits))

        # X: aplicar H antes de medir
        for q in range(qubits):
            qc_x.h(q)
        cr_x = ClassicalRegister(qubits)
        qc_x.add_register(cr_x)
        qc_x.measure(range(qubits), range(qubits))

        # Y: aplicar S† H antes de medir
        for q in range(qubits):
            qc_y.sdg(q)
            qc_y.h(q)
        cr_y = ClassicalRegister(qubits)
        qc_y.add_register(cr_y)
        qc_y.measure(range(qubits), range(qubits))

        # --- Compilación (optimización) de cada circuito para el backend ---
        pm = generate_preset_pass_manager(optimization_level=2, backend=sim)        

        compiled_z = pm.run(qc_z)
        compiled_x = pm.run(qc_x)
        compiled_y = pm.run(qc_y)

        #compiled_z = transpile(qc_z, sim, optimization_level=2)
        #compiled_x = transpile(qc_x, sim, optimization_level=2)
        #compiled_y = transpile(qc_y, sim, optimization_level=2)

        # --- Unir los compilados en una lista (o dict) ---
        compiled_circuit = [compiled_x, compiled_y, compiled_z]

    else:
        sim = AerSimulator(method="statevector", seed_simulator=33)
        
        # --- Crear copias del circuito original para medir en cada base ---
        qc_z = qc.copy()  # Para términos Z

        # --- Si es Simulation, solo aplicar rotaciones de base ---
    

        # --- Transpilar para el backend statevector ---
        compiled_z = transpile(qc_z, sim, optimization_level=0)

        # --- Unir los compilados en una lista (o dict) ---
        compiled_circuit = [compiled_z]

    # --- Construimos el tensor de signo ---
    d_t = build_sign_tensor(n_circuits = 3, n_qubits = qubits, k_degree = k)

    
    # === 5. Ejecutar la optimización VQE ===
    start_time = time.time()

    # Definir rutas para guardar resultados (CSV)
    if cunqa_str_arg == "Simulation":
        subcarpeta_csv = f"Resultados/Factorization/k_{k}/{cunqa_str_arg}/num_{N_int}/{optimizer}"
    else:
        subcarpeta_csv = f"Resultados/Factorization/k_{k}/{cunqa_str_arg}/{nqpus}qpus/num_{N_int}/{optimizer}"

    nombre_archivo_csv = f"Fact_num_{N_int}_{optimizer}_{k}.csv"    
    nombre_archivo_csv_iter =f"Fact_num_{N_int}_{optimizer}_{k}_iter.csv"
    ruta_csv = os.path.join(subcarpeta_csv, nombre_archivo_csv)
    ruta_csv_iter = os.path.join(subcarpeta_csv, nombre_archivo_csv_iter)

    # Ejecutar el proceso de optimización variacional
    result, experiment_result = run_vqe_optimization(
        sim=sim,
        n_shots=n_shots,
        alpha=alpha,
        beta=beta,
        compiled_circuit=compiled_circuit,
        N_int=N_int,
        num_p_bits=num_p_bits,
        num_q_bits=num_q_bits,
        list_size=list_size,
        d_t = d_t,
        optimizer=optimizer,
        optimizer_params= optimizer_params,
        loss_func_estimator=loss_func_estimator,
        maxiter=maxiter,
        log_csv_path=ruta_csv,
        cunqa_str=cunqa_str_arg,
        family_name = family_name
    )


        
    # === 6. Obtener y refinar la partición resultante ===
    # A partir del mapa de expectativas del último experimento, se obtiene una partición inicial
   
    min_loss = min(e["loss"] for e in experiment_result)
    best = next(
        e for e in reversed(experiment_result)
        if e["loss"] == min_loss
    )   
    exp_map = best["exp_map"]
    values = np.array([exp_map[i] for i in sorted(exp_map.keys())])
    p_bits, q_bits, p_int, q_int = alt_extract_primes(values, num_p_bits, num_q_bits)



    print(f"los números son p_primo: {p_int} y q_primo: {q_int}\n")

    num_result = p_int*q_int
    
    if num_result == N_int:
        Solution_Reached = "True"
    else:
        Solution_Reached = "False"


    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️ Tiempo total de ejecución: {elapsed:.2f} segundos")

    # === 7. Recolectar resultados finales de la optimización ===
    params = result.x             # parámetros finales del optimizador
    num_params = len(params)
    fevs = result.nfev            # número de evaluaciones de la función
    fvalue = result.fun           # valor final de la función objetivo

    print("El programa ha finalizado ✅\n")

    # Construcción del diccionario con todos los resultados de esta ejecución
    if optimizer.lower() != "cobyla":
        nit = getattr(result, "nit", None)
    else:
        nit = None

    fevs = getattr(result, "nfev", None)

    dic_resultado = {
        "compression": k,
        "qubits":qubits, 
        "elapsed_time": elapsed,
        "alpha": alpha,
        "beta": beta,
        "f_loss_value": fvalue,
        "function_evaluations": fevs,
        "Number of iterations": nit,
        "num_params": num_params,
        "params": params.tolist(),
        "Sol_Reached" : Solution_Reached,
        "p_int": int(p_int),
        "q_int": int(q_int),
        "p_bitstring": p_bits.tolist(),
        "q_bitstring": q_bits.tolist(),
        "optimizer_message": getattr(result, "message", ""),
        "optimizer_status": getattr(result, "status", ""),
    }

    print("El programa ha finalizado ✅\n")

    # === 8. Guardar resultados en archivo JSON ===
    if cunqa_str_arg == "Simulation":
        subcarpeta = f"Resultados/Factorization/k_{k}/{cunqa_str_arg}/num_{N_int}/{optimizer}"
    else:
        subcarpeta = f"Resultados/Factorization/k_{k}/{cunqa_str_arg}/{nqpus}qpus/num_{N_int}/{optimizer}"
    
    nombre_archivo = f"Fact_num_{N_int}_{optimizer}_{k}.json"

    if not os.path.isdir(subcarpeta):
        os.makedirs(subcarpeta, exist_ok=True)
    ruta_archivo = os.path.join(subcarpeta, nombre_archivo)
    append_result_to_json(ruta_archivo, dic_resultado)
    

    # === 10. Devolver resultado individual y ruta del experimento ===
    return dic_resultado, subcarpeta, ruta_csv, ruta_csv_iter
