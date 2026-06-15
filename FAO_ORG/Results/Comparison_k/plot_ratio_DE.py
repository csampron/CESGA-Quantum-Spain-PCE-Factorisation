import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURATION ===
DB_NAME = "Factorization_results.db"
OUTPUT_DIR = "Images_DE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimizers to plot.
# Use None to plot all optimizers.
#SELECTED_OPTIMIZERS = None
SELECTED_OPTIMIZERS = ["DIFFERENTIALEVOLUTION"]

# Include trivial-solution bars?
INCLUDE_TRIVIAL = True

# === READ DATABASE ===
conn = sqlite3.connect(DB_NAME)

query = """
SELECT k, numero, optimizer, ratio_exito, ratio_trivial
FROM Factorization_results
ORDER BY k, numero, optimizer
"""

df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("⚠️ The database is empty.")
    exit()

# === FILTER SELECTED OPTIMIZERS ===
if SELECTED_OPTIMIZERS is not None:
    df = df[df["optimizer"].isin(SELECTED_OPTIMIZERS)]

    if df.empty:
        print("⚠️ No data found for the selected optimizers.")
        exit()

# === GENERATE ONE IMAGE FOR EACH k ===
for k_value in sorted(df["k"].unique()):
    df_k = df[df["k"] == k_value].copy()

    # Force correct type
    df_k["numero"] = df_k["numero"].astype(int)

    pivot_success = df_k.pivot(
        index="numero",
        columns="optimizer",
        values="ratio_exito"
    ).sort_index()

    optimizers = pivot_success.columns.tolist()
    numeros = [int(n) for n in pivot_success.index]
    labels = [f"{n} ({n.bit_length()} bits)" for n in numeros]

    x = np.arange(len(numeros))

    if INCLUDE_TRIVIAL:
        pivot_trivial = df_k.pivot(
            index="numero",
            columns="optimizer",
            values="ratio_trivial"
        ).sort_index()

        width = 0.8 / (2 * len(optimizers))
    else:
        width = 0.8 / len(optimizers)

    plt.figure(figsize=(16, 5))

    for i, opt in enumerate(optimizers):

        if INCLUDE_TRIVIAL:
            offset = i * 2 * width

            plt.bar(
                x + offset,
                pivot_success[opt],
                width,
                label=f"{opt} success"
            )

            plt.bar(
                x + offset + width,
                pivot_trivial[opt],
                width,
                hatch="//",
                alpha=0.7,
                label=f"{opt} trivial"
            )

        else:
            offset = i * width

            plt.bar(
                x + offset,
                pivot_success[opt],
                width,
                label=opt
            )

    # === X AXIS ===
    if INCLUDE_TRIVIAL:
        x_center = x + width * len(optimizers)
    else:
        x_center = x + width * (len(optimizers) - 1) / 2

    plt.xticks(
        x_center,
        labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel("Factorized number (number of bits)")
    plt.ylabel("Ratio")

    if INCLUDE_TRIVIAL:
        plt.title(f"Success and trivial ratios per factorized number | k = {k_value} | Basic Approach")
    else:
        plt.title(f"Success ratio per factorized number | k = {k_value} | Basic Approach")

    plt.ylim(0, 1.1)
    plt.axhline(y=1.0, color="r", linestyle="-")

    # Legend outside the main plot
    plt.legend(
        title="Optimizer",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if INCLUDE_TRIVIAL:
        filename = f"ratio_org_success_trivial_k{k_value}.png"
    else:
        filename = f"ratio_org_success_k{k_value}.png"

    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")

print("\n📊 All images generated successfully.")