import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURATION ===
DB_NAME = "Factorization_results.db"
OUTPUT_DIR = "Images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimizers to plot.
# Use None to plot all optimizers.
SELECTED_OPTIMIZERS = None
# Example:
# SELECTED_OPTIMIZERS = ["PSO", "COBYLA"]

# === READ DATABASE ===
conn = sqlite3.connect(DB_NAME)

query = """
SELECT k, numero, optimizer, ratio_exito
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

    pivot_df = df_k.pivot(
        index="numero",
        columns="optimizer",
        values="ratio_exito"
    ).sort_index()

    optimizers = pivot_df.columns.tolist()

    # Convert index to a list of integers
    numeros = [int(n) for n in pivot_df.index]

    # Labels with bit length
    labels = [f"{n} ({n.bit_length()} bits)" for n in numeros]

    x = np.arange(len(numeros))
    width = 0.8 / len(optimizers)

    plt.figure(figsize=(10, 6))

    for i, opt in enumerate(optimizers):
        plt.bar(
            x + i * width,
            pivot_df[opt],
            width,
            label=opt
        )

    # === X AXIS ===
    plt.xticks(
        x + width * (len(optimizers) - 1) / 2,
        labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel("Factorized number (number of bits)")
    plt.ylabel("Success ratio")
    plt.title(f"Success ratio per factorized number for k = {k_value} - Original")
    plt.ylim(0, 1.5)
    plt.axhline(y=1.0, color="r", linestyle="-")
    plt.legend(title="Optimizer")

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"ratio_org_reg_success_k{k_value}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")

print("\n📊 All images generated successfully.")