import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from auxiliar.Pytest_pruebas.functions_aux import num_qubits

# ------------------------------------------------
#  Funciones matemáticas
# ------------------------------------------------

def worst_case_bits_for_primes(num_bits_4_a: int) -> tuple:
    p_bits = num_bits_4_a
    q_bits = math.ceil(num_bits_4_a / 2)
    return p_bits, q_bits




# ------------------------------------------------
#  NUEVA FUNCIÓN: calcular número de parámetros
# ------------------------------------------------

def compute_prof(m, k):

    qubits = num_qubits(m, k)

    if qubits is None:
        return None

    layers = int(np.ceil(1.5 * (qubits ** (np.floor(k / 2)))))

    num_layers = math.ceil(layers)

    from auxiliar.circuit_builder import Circuit

    qc_builder = Circuit(
        size=qubits,
        p=num_layers,
        entanglement='Taylor_efficient',
        rotation='Taylor_efficient',
        connectivity='brickwork_single_rotating',
        mode='custom'
    )

    qc_builder.compile_circuit()

    qc = qc_builder.get_circuit()

    prof = qc.depth()

    return prof


# ------------------------------------------------
#  Parámetros
# ------------------------------------------------

bits = list(range(4, 101, 4))

num_variables = 2 * np.array(bits)

ORDERS_FULL = [1, 2, 3, 4]
ORDERS_PARTIAL = [2, 3, 4]

# ------------------------------------------------
#  Paleta fija por order
# ------------------------------------------------

DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

COLORS_K = {
    1: "rebeccapurple",
    2: "steelblue",
    3: "indianred",
    4: "seagreen",
    5: "darkorange",
}


# ------------------------------------------------
#  Cálculo de resultados (AHORA PARAMS)
# ------------------------------------------------

def compute_results(order_list):

    results = {}

    for k in order_list:

        vals = []

        for m in num_variables:

            params = compute_prof(m, k)

            vals.append(params)

        results[k] = vals

    return results


results_full = compute_results(ORDERS_FULL)
results_partial = compute_results(ORDERS_PARTIAL)

# ------------------------------------------------
#  Función de plot
# ------------------------------------------------

def plot_results(order_list, results, filename, ylim=None):

    plt.figure(figsize=(9, 6))

    x = list(range(len(bits)))

    for k in order_list:

        y = []

        for q in results[k]:
            y.append(q if q is not None else float("nan"))

        plt.plot(
            x,
            y,
            marker="o",
            label=f"order={k}",
            color=COLORS_K[k]
        )

    plt.xticks(x, bits)
    if ylim!=None:
        plt.ylim(0,ylim)

    plt.xlabel("Number size (bits)")
    plt.ylabel("Depth")
    plt.title("Resulting depth vs Number size | DoTS")

    plt.legend(loc="upper left")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ------------------------------------------------
#  Generación de figuras
# ------------------------------------------------

output_dir = os.getcwd()
os.makedirs(output_dir, exist_ok=True)

plot_results(
    ORDERS_FULL,
    results_full,
    os.path.join(output_dir, "1_Depth_Fact_DoTS.png")
)

plot_results(
    ORDERS_PARTIAL,
    results_partial,
    os.path.join(output_dir, "2_Depth_Fact_DoTS.png")
)

print("✅ Imágenes generadas con número de parámetros")