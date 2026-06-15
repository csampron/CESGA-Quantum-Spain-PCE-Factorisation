import sqlite3
import json
import os
import numpy as np
import re

# =============================
# CONFIGURATION
# =============================
BASE_DIR = "/mnt/netapp1/Store_CESGA/home/cesga/falonso/z_FAO/FAO_DoTS/Results/Resultados/Factorization"
DB_NAME = "Factorization_results.db"

# Optimizers to process (empty = all)
OPTIMIZERS_TO_INCLUDE = []  # e.g., ["SACESS", "DIFFERENTIALEVOLUTION"]

# =============================
# CREATE DATABASE
# =============================
with sqlite3.connect(DB_NAME) as conn:
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS Factorization_results (
            id TEXT PRIMARY KEY,
            numero INTEGER,
            k INTEGER,
            optimizer TEXT,
            n_ejecuciones INTEGER,
            n_exitos INTEGER,
            ratio_exito REAL,
            ratio_trivial REAL,
            mean_time REAL,
            std_time REAL,
            mean_nfev REAL,
            std_nfev REAL
        )
    ''')
    conn.commit()

# =============================
# COLLECT JSON FILES
# =============================
json_files = []
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.startswith("Fact_") and file.endswith(".json"):
            json_files.append(os.path.join(root, file))

print(f"📁 Found {len(json_files)} JSON files.\n")

# =============================
# PROCESS JSON FILES AND SAVE TO DB
# =============================
conn = sqlite3.connect(DB_NAME)
c = conn.cursor()

for ruta_json in json_files:
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        resultados = data.get("resultados", [])
        if not resultados:
            print(f"⚠️ No results found in: {ruta_json}")
            continue

        # Extract factored number and k value from path
        m_num = re.search(r"num_(\d+)", ruta_json)
        numero = int(m_num.group(1)) if m_num else None

        m_k = re.search(r"k_(\d+)", ruta_json)
        k_value = int(m_k.group(1)) if m_k else None

        # Extract optimizer from path
        optimizer = os.path.basename(os.path.dirname(ruta_json))

        # Filter by optimizer if specified
        if OPTIMIZERS_TO_INCLUDE and optimizer not in OPTIMIZERS_TO_INCLUDE:
            continue

        # Keep only correct and non-trivial solutions
        filtered_results = [
            r for r in resultados
            if str(r.get("Sol_Reached", "False")).lower() == "true"
               and int(r.get("p_int", 1)) != 1
               and int(r.get("q_int", 1)) != 1
        ]

        n_ejecuciones = len(filtered_results)
        n_exitos = n_ejecuciones  # only counting filtered results
        ratio_exito = n_exitos / len(resultados) if resultados else 0.0

        # Count trivial solutions
        triviales = sum(
            1 for r in resultados
            if int(r.get("p_int", 1)) == 1 or int(r.get("q_int", 1)) == 1
        )

        ratio_trivial = triviales / len(resultados) if resultados else 0.0

        # Extract metrics
        tiempos = [r.get("elapsed_time", 0.0) for r in filtered_results]
        nfevs = [r.get("function_evaluations", 0) for r in filtered_results]

        mean_time = float(np.mean(tiempos)) if tiempos else 0.0
        std_time = float(np.std(tiempos)) if tiempos else 0.0

        mean_nfev = float(np.mean(nfevs)) if nfevs else 0.0
        std_nfev = float(np.std(nfevs)) if nfevs else 0.0

        unique_id = f"k{k_value}_n{numero}_{optimizer}"

        params = (
            unique_id,
            numero,
            k_value,
            optimizer,
            n_ejecuciones,
            n_exitos,
            ratio_exito,
            ratio_trivial,
            mean_time,
            std_time,
            mean_nfev,
            std_nfev
        )

        c.execute('''
            INSERT OR REPLACE INTO Factorization_results
            (id, numero, k, optimizer, n_ejecuciones, n_exitos,
             ratio_exito, ratio_trivial, mean_time, std_time,
             mean_nfev, std_nfev)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', params)

        print(
            f"✅ k={k_value} | number={numero} | {optimizer} | "
            f"Runs={n_ejecuciones} | Time={mean_time:.3f}s ± {std_time:.3f}s | "
            f"NFEV={mean_nfev:.1f} ± {std_nfev:.1f}"
        )

    except Exception as e:
        print(f"❌ Error processing {ruta_json}: {e}")

conn.commit()
conn.close()

print("\n📊 Factorization database successfully updated.")