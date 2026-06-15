
from typing import Tuple
import math
from itertools import combinations
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


def solve_for_k(m: float, k: int, max_n: int = 10000, tol: float = 1e-6):
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