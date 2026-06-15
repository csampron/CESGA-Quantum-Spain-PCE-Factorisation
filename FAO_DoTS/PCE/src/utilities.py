### ========================================================= ###
### Módulo: utilities
### ========================================================= ###
###
### Este módulo proporciona funciones para la optimización variacional
### (VQE) de ansatz cuánticos y manejo de resultados de evaluaciones.
###
### ========================================================= ###

from scipy.optimize import minimize, differential_evolution, brute, OptimizeResult


from src import pso_simple
from src import qpso_simple

import numpy as np
import csv
import os
from functools import partial


def run_vqe_optimization(
    sim,
    n_shots,
    alpha,
    beta,
    compiled_circuit,
    N_int,
    num_a_bits,
    num_b_bits,
    list_size,
    d_t,
    optimizer,
    optimizer_params=None,
    loss_func_estimator=None,
    maxiter=1000,
    log_csv_path=None,
    cunqa_str="None",
    family_name=None
):
    """
    Ejecuta una optimización variacional (VQE) sobre un ansatz cuántico dado.

    Retorna:
    --------
    result : OptimizeResult
        Resultado del optimizador.
    experiment_result : list
        Lista con los resultados de la función de pérdida en cada evaluación.
    """

    experiment_result = []
    cost_history = []
    iteration_costs = []

    # =========================================================
    # CALLBACKS
    # =========================================================

    def callback(xk):
        value = loss_func(xk)
        iteration_costs.append(value)

    def callback_de(intermediate_result):
        iteration_costs.append(intermediate_result.fun)
        return None
    
    def callback_de_indicator(intermediate_result):
        iteration_costs.append(intermediate_result.fun)

        x_best = intermediate_result.x
        indicator_value = base_loss_func(x_best)

        if indicator_value == 0.0:
            print("🎯 Se alcanzó solución física (indicador = 0.0)")
            return True

        return False

    def callback_spsa(nfev, x, fx, stepsize, accepted):
        cost_history.append(fx)

    # =========================================================
    # LOSS FUNCTION
    # =========================================================

    def make_loss_functions():
        cache = {"last_x": None, "total": None, "base": None}

        def loss_func(x):
            total, base = loss_func_estimator(
                x=x,
                alpha=alpha,
                beta=beta,
                ansatz=compiled_circuit,
                sim=sim,
                N_int=N_int,
                num_a_bits=num_a_bits,
                num_b_bits=num_b_bits,
                list_size=list_size,
                num_qubits=compiled_circuit[0].num_qubits,
                d_t=d_t,
                n_shots=n_shots,
                experiment_result=experiment_result,
                CUNQA=cunqa_str,
                family_name=family_name
            )
            cache["last_x"] = tuple(x)
            cache["total"] = total
            cache["base"] = base
            cost_history.append(total)
            return total

        def base_loss_func(x):
            if cache["last_x"] == tuple(x):
                return cache["base"]
            else:
                _, base = loss_func_estimator(
                    x=x,
                    alpha=alpha,
                    beta=beta,
                    ansatz=compiled_circuit,
                    sim=sim,
                    N_int=N_int,
                    num_a_bits=num_a_bits,
                    num_b_bits=num_b_bits,
                    list_size=list_size,
                    num_qubits=compiled_circuit[0].num_qubits,
                    d_t=d_t,
                    n_shots=n_shots,
                    experiment_result=experiment_result,
                    CUNQA=cunqa_str,
                    family_name=family_name
                )
                return base

        return loss_func, base_loss_func

    loss_func, base_loss_func = make_loss_functions()

    # =========================================================
    # PARAMETROS INICIALES
    # =========================================================

    rng_default = np.random.default_rng(33)
    initial_params = rng_default.random(len(compiled_circuit[0].parameters)) * 2 * np.pi

    # =========================================================
    # OPTIMIZADORES
    # =========================================================

    if isinstance(optimizer, str):

        optimizer_lower = optimizer.lower()

        default_optimizers = {


            "powell": {
                "method": "Powell",
                "options": {
                    "maxiter": maxiter,
                    "maxfev": 50000,
                    "xtol": 1e-9,
                    "ftol": 1e-9,
                    "disp": True,
                    "return_all": True
                },
                "callback": callback
            },

            "cobyla": {
                "method": "COBYLA",
                "options": {
                    "maxiter": maxiter,
                    "rhobeg": 1.0,
                    "tol": 1e-9,
                    "disp": False
                },
                "callback": callback
            },

            "bfgs": {
                "method": "BFGS",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "gtol": 1e-3
                },
                "callback": callback
            },

            "lbfgsb": {
                "method": "L-BFGS-B",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "ftol": 1e-6,
                    "gtol": 1e-3,
                    "eps": 1e-3
                },
                "callback": callback
            },

            "tnc": {
                "method": "TNC",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "ftol": 1e-8,
                    "xtol": 1e-8,
                    "gtol": 1e-4
                },
                "callback": callback
            },

            "slsqp": {
                "method": "SLSQP",
                "options": {
                    "disp": True,
                    "maxiter": maxiter,
                    "ftol": 1e-8,
                    "eps": 1e-6
                },
                "callback": callback
            },

            "differentialevolution": {
                "kwargs": {
                    "strategy": "best1exp",
                    "maxiter": maxiter,
                    "popsize": 10,
                    "tol": 0,
                    "atol": 0,
                    "mutation": (0.5, 1),
                    "recombination": 0.7,
                    "disp": True,
                    "polish": False,
                    "init": "halton"
                }
            },

            "pso": {
                "kwargs": {
                    "num_particles": 100,
                    "maxiter": maxiter,
                    "verbose": True
                }
            },

            "qdpso": {
                "kwargs": {
                    "num_particles": 100,
                    "maxiter": maxiter,
                    "verbose": True,
                    "g": 0.7   # parámetro de dispersión cuántica
                }
            },

            "brute": {}
        }

        if optimizer_lower not in default_optimizers:
            raise ValueError(f"Optimizador desconocido: {optimizer}")

        opt_config = default_optimizers[optimizer_lower]

        # =====================================================
        # Sobrescribir parámetros personalizados
        # =====================================================

        if optimizer_params:
            print(f"⚡ Usando parámetros personalizados para {optimizer.upper()}")

            if optimizer_lower in ["differentialevolution", "pso", "qdpso"]:
                opt_config["kwargs"].update(optimizer_params)
            else:
                opt_config["options"].update(optimizer_params)

        else:
            print(f"⚡ Usando parámetros por defecto para {optimizer.upper()}")

        # =====================================================
        # Ejecutar optimizador
        # =====================================================

        if optimizer_lower == "pso":

            bounds = [(0, 2*np.pi)] * len(initial_params)

            # Ejecutar PSO clásico
            best_value, best_position, iteration_costs = pso_simple.minimize(
                loss_func,                 # función que se optimiza
                base_loss_func,            # función para criterio de parada
                initial_params,
                bounds,
                **opt_config["kwargs"]
            )

            # Asegurar que sea array numpy
            best_position = np.atleast_1d(best_position)

            # Calcular número de evaluaciones
            num_particles = opt_config["kwargs"].get("num_particles", 20)
            maxiter = opt_config["kwargs"].get("maxiter", 1000)
            
            nfev = len(cost_history)
            nit = int(np.ceil(nfev / num_particles))

            # Crear OptimizeResult compatible con SciPy
            result = OptimizeResult(
                x=best_position,
                fun=best_value,
                nfev=nfev,
                nit=nit,
                success=True,
                message="PSO optimization finished."
            )

        elif optimizer_lower == "qdpso":

            bounds = [(0, 2*np.pi)] * len(initial_params)

            # Ejecutar QDPSO
            best_value, best_position, history = qpso_simple.minimize_qdpso(
                loss_func=loss_func,          # con regularización
                base_loss_func=base_loss_func,      # sin regularización
                x0=initial_params,
                bounds=bounds,
                **opt_config["kwargs"]
            )

            # Calcular número de evaluaciones
            best_position = np.atleast_1d(best_position)
            num_particles = opt_config["kwargs"].get("num_particles", 20)
            maxiter = opt_config["kwargs"].get("maxiter", 1000)
            
            nfev = len(cost_history)
            nit = int(np.ceil(nfev / num_particles))

            result = OptimizeResult(
                x=best_position,
                fun=best_value,
                nfev=nfev,
                nit=nit,
                success=True,
                message="QDPSO optimization finished."
            )

        elif optimizer_lower == "differentialevolution":

            result = differential_evolution(
                loss_func,
                bounds=[(0, 2*np.pi)] * len(initial_params),
                callback=callback_de_indicator,
                **opt_config["kwargs"]
            )

        elif optimizer_lower == "brute":

            result = brute(
                loss_func,
                ranges=[(0, 2*np.pi)] * len(initial_params)
            )

        else:

            result = minimize(
                loss_func,
                initial_params,
                method=opt_config["method"],
                options=opt_config["options"],
                callback=opt_config["callback"]
            )

    else:
        result = optimizer.minimize(fun=loss_func, x0=initial_params)

    # =========================================================
    # Guardar CSV si se solicita
    # =========================================================

    import fcntl
    import os
    import csv

    if log_csv_path:
        # Asegurarse de que exista la carpeta
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

        # Abrir el CSV con lock
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            # Bloqueo exclusivo
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.writer(f)
            writer.writerow(["iteracion", "valor_coste"])
            for i, val in enumerate(cost_history):
                writer.writerow([i, val])
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)  # desbloquear

        iter_csv_path = log_csv_path.replace(".csv", "_iter.csv")
        with open(iter_csv_path, "w", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.writer(f)
            writer.writerow(["iteracion", "valor_coste"])
            for i, val in enumerate(iteration_costs):
                writer.writerow([i, val])
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    return result, experiment_result