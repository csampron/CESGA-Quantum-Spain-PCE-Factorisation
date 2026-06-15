### ========================================================= ###
### Módulo: Quantum Graph Encoding & Local Refinement
### ========================================================= ###
###
### Este módulo proporciona funciones para:
### 1. Calcular el número de qubits necesarios para un problema PCE.
### 2. Resolver ecuaciones cuadráticas o generales de combinatoria para el mapeo de variables.
### 3. Construir codificaciones de Hamiltonianos mediante operadores de Pauli correlacionados.
### 4. Generar particiones iniciales a partir de mapas de expectativas (exp_map).
### 5. Refinamiento local de particiones para mejorar el tamaño del corte (max-cut).
###
### ========================================================= ###

from typing import Tuple
import math
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def num_qubits(num_variables, order_compression):
    """
    Calcula el número de qubits necesarios para un problema dado
    según el número de variables y el orden de compresión PCE.

    Parámetros:
    -----------
    num_variables : int
        Número de variables del problema (por ejemplo, número de nodos en un grafo)
    order_compression : int
        Orden de compresión de PCE (por ejemplo 2 para cuadrático)

    Retorna:
    --------
    int
        Número de qubits requeridos
    """
    if order_compression == 1:
        # Para el caso cuadrático se resuelve una ecuación cuadrática
        n_ceil = math.ceil(num_variables/3)

        if 3*math.comb(n_ceil-1, order_compression) >= num_variables:
            qubits = n_ceil-1
        else:
            qubits = n_ceil

    elif order_compression == 2:
        # Para el caso cuadrático se resuelve una ecuación cuadrática
        n_ceil = math.ceil(max(solve_quadratic(1, -1, -2 / 3 * num_variables)))

        if 3*math.comb(n_ceil-1, order_compression) >= num_variables:
            qubits = n_ceil-1
        else:
            qubits = n_ceil
    else:
        # Para orden mayor se usa un método general
        n_real = solve_for_k(num_variables, order_compression)
        
        n_ceil = int(math.ceil(n_real))
        
        if 3*math.comb(n_ceil-1, order_compression) >= num_variables:
            qubits = n_ceil-1
        else:
            qubits = n_ceil
            
    return qubits

def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Resuelve la ecuación cuadrática a*x^2 + b*x + c = 0.

    Retorna:
    --------
    tuple
        Las dos soluciones (reales o complejas)
    """
    discriminant = b ** 2 - 4 * a * c
    if discriminant >= 0:
        x_1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x_2 = (-b - math.sqrt(discriminant)) / (2 * a)
    else:
        x_1 = complex((-b / (2 * a)), math.sqrt(-discriminant) / (2 * a))
        x_2 = complex((-b / (2 * a)), -math.sqrt(-discriminant) / (2 * a))
    return x_1, x_2


def solve_for_k(m: float, k: int, max_n: int = 100, tol: float = 1e-6):
    """
    Resuelve m = 3 * comb(n, k) para n, dado k y m.
    Usa log-gamma para evitar overflow con números grandes.
    """
    def comb_continuous(n, k):
        # Usando log-gamma para evitar overflow
        log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        return math.exp(log_comb)
    
    def f(n):
        return 3 * comb_continuous(n, k) - m
    
    # Búsqueda del intervalo inicial
    a, b = k, max_n
    if f(a) > 0:
        return None
    
    while f(b) < 0 and b < 1e6:
        b *= 2  # ampliar el intervalo si es necesario
    
    # Búsqueda binaria
    for _ in range(100):
        mid = (a + b) / 2
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if fmid < 0:
            a = mid
        else:
            b = mid
    return (a + b) / 2


### ========================================================= ###
### FUNCIONES PARA TRATAMIENTOS BINARIOS
### ========================================================= ###

def bits_required(a: int) -> int:
    """
    Calcula el número de bits necesarios para representar
    un entero positivo en binario.

    Equivale a: floor(log2(a)) + 1.
    Utiliza la función incorporada bit_length de Python.

    Parámetros:
    -----------
    a : int
        Entero positivo a representar en binario

    Retorna:
    --------
    int
        Número de bits necesarios para representar a
    """
    num_bits = a.bit_length()
    return num_bits


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


def int_to_bits(int_num: int, bits_num: int) -> np.ndarray:
    """
    Convierte un entero en su representación binaria
    como un vector de bits de longitud fija.

    El orden del vector es:
    [LSB, ..., MSB]

    Parámetros:
    -----------
    int_num : int
        Entero a convertir
    bits_num : int
        Número total de bits del vector resultante

    Retorna:
    --------
    np.ndarray
        Vector de bits (0/1) de longitud bits_num
    """
    bits_vec = ((int_num >> np.arange(bits_num)) & 1).astype(np.int8)
    return bits_vec


def bits_to_int(bits_vec: np.ndarray) -> int:
    """
    Convierte un vector de bits en el entero correspondiente.

    Se asume que el orden del vector es:
    [LSB, ..., MSB]

    Parámetros:
    -----------
    bits_vec : np.ndarray
        Vector de bits en orden LSB → MSB

    Retorna:
    --------
    int
        Entero representado por el vector de bits
    """
    int_num = int(np.dot(bits_vec, 1 << np.arange(np.size(bits_vec))))
    return int_num


def ising_to_binary(ising_vec: np.array) -> np.array:
    """
    Convierte un vector de variables Ising {-1, +1}
    en un vector binario {0, 1}.

    La conversión se realiza como:
    b = (1 + s) / 2

    Parámetros:
    -----------
    ising_vec : np.array
        Vector de variables Ising

    Retorna:
    --------
    np.array
        Vector binario correspondiente
    """
    bin_vec = (1 + ising_vec) / 2
    return bin_vec


def kron_delta(i: int, j: int) -> int:
    """
    Implementa la delta de Kronecker.

    Devuelve 1 si i == j y 0 en caso contrario.

    Parámetros:
    -----------
    i : int
        Primer índice
    j : int
        Segundo índice

    Retorna:
    --------
    int
        1 si i == j, 0 en otro caso
    """
    return int(i == j)


def alt_extract_primes(node_exp_map, num_p_bits, num_q_bits):
    """
    Extrae los vectores binarios y los valores enteros
    asociados a dos factores (p y q) a partir de un
    mapa de expectativas Ising.

    El vector Ising se binariza y luego se separa en:
    - x: bits del primo p
    - y: bits del primo q

    El bit menos significativo de cada primo se fija a 1.

    Parámetros:
    -----------
    node_exp_map : array-like
        Vector de expectativas Ising por nodo
    num_p_bits : int
        Número de bits asignados al primo p
    num_q_bits : int
        Número de bits asignados al primo q

    Retorna:
    --------
    tuple
        (x_vec, y_vec, x, y) donde:
        x_vec : np.ndarray
            Vector de bits del primo p
        y_vec : np.ndarray
            Vector de bits del primo q
        x : int
            Valor entero del primo p
        y : int
            Valor entero del primo q
    """
    bin_vec = ising_to_binary(np.sign(node_exp_map))

    x_vec = np.ones(num_p_bits)
    y_vec = np.ones(num_q_bits)

    x_vec = bin_vec[:num_p_bits]

    y_vec = bin_vec[num_p_bits:]

    x = bits_to_int(x_vec)
    y = bits_to_int(y_vec)

    return x_vec, y_vec, x, y



def extract_factors(a_int, b_int, N_int):
    """
    Calcula los factores p y q de N usando gcd(a-b, N)

    Arguments:
    a_int: entero a
    b_int: entero b
    N_int: entero a factorizar (semiprimo)

    Returns:
    p, q: factores encontrados si son no triviales, o (None, None) si no se encontró
    """
    import math

    aux = a_int - b_int

    # gcd(a-b, N)
    g = math.gcd(aux, N_int)

    # Comprobar que sea factor no trivial
    p = g
    q = N_int // g

    p_bits = int_to_bits(p, math.ceil(np.log2(p)))
    q_bits = int_to_bits(q, math.ceil(np.log2(q)))

    return p, q, p_bits, q_bits