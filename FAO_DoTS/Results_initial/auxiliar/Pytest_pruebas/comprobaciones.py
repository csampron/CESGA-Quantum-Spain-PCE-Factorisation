import math
import pytest
from functions_aux import num_qubits, solve_for_k

# -----------------------------
# Test de la función solve_for_k
# -----------------------------
@pytest.mark.parametrize("k,n", [
    (2,5),
    (2,10),
    (3,6),
    (3,8),
    (4,9),
])
def test_solve_for_k_equation(k, n):
    """
    Comprueba que solve_for_k devuelve una raíz real cercana al valor entero n.
    La función solve_for_k devuelve la raíz real de la ecuación:
        3 * comb(n_est, k) = m
    """
    m = 3 * math.comb(n, k)

    n_est = solve_for_k(m, k)

    # La raíz real debe estar muy cerca del entero esperado
    assert abs(n_est - n) < 1e-3


# -----------------------------------------
# Test de num_qubits basado en inecuación
# -----------------------------------------
@pytest.mark.parametrize("k,n", [
    (2,5),
    (2,10),
    (3,6),
    (3,8),
    (4,9),
])
def test_num_qubits_inequality_general(k, n):
    """
    Comprueba que num_qubits devuelve el mínimo entero q
    tal que 3*comb(q, k) >= m. 
    Verifica además que q-1 no cumpliría la desigualdad.
    """
    m = 3 * math.comb(n, k)

    q = num_qubits(m, k)

    # q debe cumplir la desigualdad
    assert 3 * math.comb(q, k) >= m
    # q-1 no cumpliría la desigualdad (solo si q > k)
    if q > k:
        assert 3 * math.comb(q-1, k) < m


# -----------------------------
# Test de monotonicidad de num_qubits
# -----------------------------
def test_num_qubits_monotonic():
    """
    Verifica que la función num_qubits es monótona creciente
    respecto a m, es decir, si aumentamos m, el número de qubits
    no disminuye.
    """
    previous = num_qubits(1, 2)

    for m in range(2, 100):
        q = num_qubits(m, 2)
        # Cada nuevo valor debe ser al menos igual al anterior
        assert q >= previous
        previous = q


# -----------------------------
# Test de robustez con múltiples casos
# -----------------------------
def test_random_cases():
    """
    Comprueba múltiples valores de n y k para asegurar que
    num_qubits devuelve el mínimo entero correcto
    según la inecuación 3*comb(q, k) >= m
    """
    for k in range(2,6):
        for n in range(k+1,15):
            m = 3 * math.comb(n, k)
            q = num_qubits(m, k)
            # Verifica la propiedad de inecuación mínima
            assert 3 * math.comb(q, k) >= m
            if q > k:
                assert 3 * math.comb(q-1, k) < m