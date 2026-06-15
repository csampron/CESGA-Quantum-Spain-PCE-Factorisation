import os
import math
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from auxiliar.Pytest_pruebas.functions_aux import num_qubits

# ------------------------------------------------
#  Funciones matemáticas
# ------------------------------------------------
def worst_case_bits_for_primes(num_bits_4_a: int) -> tuple:
    """
    Calcula el peor caso para la distribución del número
    de bits de dos factores primos p y q, dado el número
    de bits del producto a = p * q.

    Se asume:
    - p tiene num_bits_4_a bits
    - q tiene ceil(num_bits_4_a / 2) bits

    Parámetros:
    -----------
    num_bits_4_a : int
        Número de bits del entero compuesto a

    Retorna:
    --------
    tuple
        (m, n) donde:
        m : int
            Número de bits del primo p
        n : int
            Número de bits del primo q
    """
    p_bits = num_bits_4_a
    q_bits = math.ceil(num_bits_4_a / 2)
    return p_bits, q_bits


# ------------------------------------------------
#  Parámetros
# ------------------------------------------------

bits = [48, 80, 100, 762, 768, 795, 829, 2048, 4096] 

num_p_bits, num_q_bits = zip(
    *[worst_case_bits_for_primes(b) for b in bits]
)

num_p_bits = list(num_p_bits)
num_q_bits = list(num_q_bits)

num_variables = [p + q for p, q in zip(num_p_bits, num_q_bits)]

ORDERS_FULL = [1, 2, 3, 4, 5]
ORDERS_PARTIAL = [2, 3, 4, 5]

# ------------------------------------------------
#  Paleta fija por order (CLAVE)
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
#  Cálculo de resultados
# ------------------------------------------------

def compute_results(order_list):
    results = {}
    for k in order_list:
        vals = []
        for n in num_variables:
            q = num_qubits(n, k)
            vals.append(q)
        results[k] = vals
    return results


results_full = compute_results(ORDERS_FULL)
results_partial = compute_results(ORDERS_PARTIAL)

# ------------------------------------------------
#  Función de plot (colores garantizados)
# ------------------------------------------------

def plot_results(order_list, results, filename, ylim=None):

    plt.figure(figsize=(9, 6))

    x = list(range(len(bits)))   # posiciones equiespaciadas

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

    plt.xticks(x, bits)  # etiquetas reales
    if ylim!=None:
        plt.ylim(0,ylim)

    plt.xlabel("RSA size (bits)")
    plt.ylabel("Qubits required")
    plt.title("Qubits required vs RSA size | Basic Approach")
    plt.legend(loc="upper left")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ------------------------------------------------
#  Generación de figuras
# ------------------------------------------------

output_dir = "/mnt/netapp1/Store_CESGA/home/cesga/falonso/z_FAO/Resources"
os.makedirs(output_dir, exist_ok=True)

plot_results(
    ORDERS_FULL,
    results_full,
    os.path.join(output_dir, "1_Qubits_Fact_org_challenges.png")
)

plot_results(
    ORDERS_PARTIAL,
    results_partial,
    os.path.join(output_dir, "2_Qubits_Fact_org_challenges.png")
)

print("✅ Imágenes generadas con colores coherentes")
