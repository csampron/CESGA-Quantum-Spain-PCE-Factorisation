import sqlite3
import json
import os
import numpy as np
import re

# === CONFIGURATION ===
BASE_DIR = "/mnt/netapp1/Store_CESGA/home/cesga/falonso/z_FAO/FAO_ORG/PCE_parallel/Resultados/Factorization"
DB_NAME = "Factorization_results.db"

# === CREATE DATABASE ===
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
        tiempo_medio REAL,
        media_loss REAL,
        desviacion_loss REAL
    )
    ''')
    conn.commit()

# === COLLECT JSON FILES ===
json_files = []

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.startswith("Fact_") and file.endswith(".json"):
            json_files.append(os.path.join(root, file))

print(f"📁 Found {len(json_files)} JSON files.\n")

# === PROCESS FILES ===
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

        # === Extract factored number from path ===
        m_num = re.search(r"num_(\d+)", ruta_json)
        numero = int(m_num.group(1)) if m_num else None

        # === Extract k value from path ===
        m_k = re.search(r"k_(\d+)", ruta_json)
        k_value = int(m_k.group(1)) if m_k else None

        # === Extract optimizer from path ===
        optimizer = os.path.basename(os.path.dirname(ruta_json))

        # === Metrics ===
        n_ejecuciones = len(resultados)

        tiempos = [r.get("elapsed_time", 0.0) for r in resultados]
        losses = [r.get("f_loss_value", 0.0) for r in resultados]

        tiempo_medio = float(np.mean(tiempos))
        media_loss = float(np.mean(losses))
        desviacion_loss = float(np.std(losses))

        # === Successes (correct non-trivial solutions) ===
        exitos = sum(
            1 for r in resultados
            if (
                str(r.get("Sol_Reached", "False")).lower() == "true"
                and int(r.get("p_int", 1)) != 1
                and int(r.get("q_int", 1)) != 1
            )
        )

        ratio_exito = exitos / n_ejecuciones if n_ejecuciones > 0 else 0.0

        # === Trivial solutions ===
        triviales = sum(
            1 for r in resultados
            if (
                int(r.get("p_int", 1)) == 1
                or int(r.get("q_int", 1)) == 1
            )
        )

        ratio_trivial = triviales / n_ejecuciones if n_ejecuciones > 0 else 0.0

        unique_id = f"k{k_value}_n{numero}_{optimizer}"

        params = (
            unique_id,
            numero,
            k_value,
            optimizer,
            n_ejecuciones,
            exitos,
            ratio_exito,
            ratio_trivial,
            tiempo_medio,
            media_loss,
            desviacion_loss
        )

        c.execute('''
            INSERT OR REPLACE INTO Factorization_results
            (id, numero, k, optimizer, n_ejecuciones, n_exitos,
             ratio_exito, ratio_trivial, tiempo_medio, media_loss, desviacion_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', params)

        print(
            f"✅ k={k_value} | number={numero} | {optimizer} | "
            f"Successes={exitos}/{n_ejecuciones} | "
            f"Trivial={triviales}/{n_ejecuciones} | "
            f"Success ratio={ratio_exito:.3f} | "
            f"Trivial ratio={ratio_trivial:.3f}"
        )

    except Exception as e:
        print(f"❌ Error processing {ruta_json}: {e}")

conn.commit()
conn.close()

print("\n📊 Factorization database successfully updated.")