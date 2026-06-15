#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import ceil, log2

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_DIR = "/mnt/netapp1/Store_CESGA/home/cesga/falonso/z_FAO/FAO_DoTS/Results/Resultados/Factorization"
OUTPUT_DIR = "./Images_comparative_nfevs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTIMIZERS = ["DIFFERENTIALEVOLUTION"]

# 🔹 FLAGS
APPLY_FIT = False
PLOT_COMBINATIONS = True

LIMIT_X_RANGE = False
X_MIN = 9
X_MAX = 37

# 🔹 NEW: also generate logarithmic-scale plots
GENERATE_LOG_PLOTS = True

# -----------------------------
# MODELS
# -----------------------------
def linear(x, a, b):
    return a * x + b

def power_law(x, a, b):
    return a * np.power(x, b)

def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

def gnfs_like(x, a, b):
    return a * np.exp(
        b * np.power(x, 1 / 3) * np.power(np.log(x), 2 / 3)
    )

def fit_models(x, y):
    models = {
        "linear": linear,
        "power": power_law,
        "exp": exponential,
        "log": logarithmic,
        "gnfs": gnfs_like
    }

    results = []

    for name, func in models.items():
        try:
            if len(x) < 2:
                continue

            if name == "log" and np.any(x <= 0):
                continue

            popt, _ = curve_fit(func, x, y, maxfev=20000)
            y_pred = func(x, *popt)

            mse = np.mean((y - y_pred) ** 2)

            results.append((name, func, popt, mse))

        except Exception:
            continue

    results.sort(key=lambda r: r[3])

    return results

# -----------------------------
# PLOTTING FUNCTION
# -----------------------------
def plot_and_fit(x, y, yerr, xlabel, filename, k, optimizer, use_log=False):

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    x, y, yerr = x[mask], y[mask], yerr[mask]

    if LIMIT_X_RANGE:
        mask_range = (x >= X_MIN) & (x <= X_MAX)
        x, y, yerr = x[mask_range], y[mask_range], yerr[mask_range]

    if len(x) < 2:
        print(f"[WARNING] Not enough data after filtering (k={k}, {optimizer})")
        return

    plt.figure(figsize=(12, 6))
    plt.errorbar(x, y, yerr=yerr, fmt='o', label="Data ± std", zorder=3)

    # -----------------------------
    # 2^n CURVE
    # -----------------------------
    if PLOT_COMBINATIONS:
        x_comb = np.linspace(min(x), max(x), 300)
        y_comb = np.power(2, x_comb)

        plt.plot(
            x_comb,
            y_comb,
            label=r"$2^n$",
            linestyle=':',
            linewidth=2
        )

    # -----------------------------
    # MODEL FITTING
    # -----------------------------
    if APPLY_FIT:

        results = fit_models(x, y)

        print(
            f"\n📊 Model ranking for {xlabel} "
            f"(k={k}, {optimizer}, log={use_log}):"
        )

        for name, _, _, mse in results:
            print(f"  {name:7s} -> MSE={mse:.3e}")

        x_fit = np.linspace(min(x), max(x), 300)

        if results:
            name, func, popt, mse = results[0]

            y_fit = func(x_fit, *popt)

            if name == "linear":
                eq = f"y = {popt[0]:.3e}x + {popt[1]:.3e}"

            elif name == "power":
                eq = rf"y = {popt[0]:.3e}x$^{{{popt[1]:.3e}}}$"

            elif name == "exp":
                eq = f"y = {popt[0]:.3e}·exp({popt[1]:.3e}x)"

            elif name == "log":
                eq = f"y = {popt[0]:.3e}·ln(x) + {popt[1]:.3e}"

            elif name == "gnfs":
                eq = "GNFS-like"

            else:
                eq = name

            plt.plot(
                x_fit,
                y_fit,
                "--",
                label=f"{name}: {eq} (MSE={mse:.2e})"
            )

    # -----------------------------
    # LOG SCALE (OPTIONAL)
    # -----------------------------
    if use_log:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("Number of function evaluations (Nfev)")
    plt.title(f"Nfev vs Search Space Size | {optimizer} | k={k}")

    plt.grid(True)
    plt.legend(loc="upper left")

    if LIMIT_X_RANGE:
        plt.xlim(X_MIN, X_MAX)

    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"✔ {filename} generated")

# -----------------------------
# READ AND PROCESS JSON FILES
# -----------------------------
data_by_k = {}

for k_dir in os.listdir(INPUT_DIR):

    if not k_dir.startswith("k_"):
        continue

    k = int(k_dir.split("_")[1])

    sim_base_dir = os.path.join(
        INPUT_DIR,
        k_dir,
        "Simulation"
    )

    if not os.path.isdir(sim_base_dir):
        continue

    for optimizer in OPTIMIZERS:

        data_by_k[(k, optimizer)] = []

        for num_dir in os.listdir(sim_base_dir):

            if not num_dir.startswith("num_"):
                continue

            try:
                num = int(num_dir.split("_")[1])

            except Exception:
                continue

            num_bits = ceil(log2(num))

            optimizer_dir = os.path.join(
                sim_base_dir,
                num_dir,
                optimizer
            )

            if not os.path.isdir(optimizer_dir):
                continue

            for fname in os.listdir(optimizer_dir):

                if not fname.endswith(".json"):
                    continue

                fpath = os.path.join(optimizer_dir, fname)

                try:
                    with open(fpath) as f:
                        data = json.load(f)

                    for r in data.get("resultados", []):

                        nfev = r.get("function_evaluations")

                        if (
                            nfev is not None
                            and str(r.get("Sol_Reached", "False")).lower() == "true"
                            and int(r.get("p_int", 1)) != 1
                            and int(r.get("q_int", 1)) != 1
                        ):
                            data_by_k[(k, optimizer)].append(
                                (num_bits, nfev)
                            )

                except Exception as e:
                    print(f"❌ Error reading {fpath}: {e}")

                print(
                    f"Processed: k={k}, num={num}, optimizer={optimizer}"
                )

# -----------------------------
# GENERATE PLOTS FOR EACH k
# -----------------------------
for (k, optimizer), values in data_by_k.items():

    if not values:
        print(f"⚠ No data available for k={k}, optimizer={optimizer}")
        continue

    bits_dict = {}

    for b, v in values:
        bits_dict.setdefault(b, []).append(v)

    x_bits = []
    y_mean = []
    y_std = []

    for b in sorted(bits_dict.keys()):

        vals = np.array(bits_dict[b])

        if len(vals) == 0:
            continue

        x_bits.append(b)
        y_mean.append(np.mean(vals))
        y_std.append(np.std(vals))

    x_bits = np.array(x_bits)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)

    print(f"\n📌 k={k}, optimizer={optimizer}")
    print("bits:", x_bits)
    print("mean nfev:", y_mean)
    print("std:", y_std)

    # 🔹 Standard plot
    filename = f"nfev_k{k}_{optimizer}_num_bits.png"

    plot_and_fit(
        x_bits,
        y_mean,
        y_std,
        "Number of bits (n)",
        filename,
        k,
        optimizer,
        use_log=False
    )

    # 🔹 Logarithmic-scale plot (if enabled)
    if GENERATE_LOG_PLOTS:

        filename_log = f"nfev_k{k}_{optimizer}_num_bits_log.png"

        plot_and_fit(
            x_bits,
            y_mean,
            y_std,
            "Number of bits (n)",
            filename_log,
            k,
            optimizer,
            use_log=True
        )

print("\n📊 NFEV analysis completed.")