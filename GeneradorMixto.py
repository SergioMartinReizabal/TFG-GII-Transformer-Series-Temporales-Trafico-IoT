#!/usr/bin/env python
# intercalar_total_n.py ----------------------------------------------
"""
Genera un CSV con EXACTAMENTE TOTAL_VENTANAS ventanas, alternando
bloques de la secuencia indicada. No repite ninguna ventana dentro de un
mismo CSV origen: si se acaba, lanza error para que añadas más ficheros.

columnas requeridas en los CSV de origen:
    - Timestamp
    - Ventana_Inicio
    - Label (se sobreescribe)
"""

from pathlib import Path
from datetime import timedelta
import pandas as pd

# ---------- CONFIGURA -------------------------------------------------
SECUENCIA = [
    # etiqueta     csv_origen              ventanas_por_bloque
    ("Benigno",   "/Users/sergiomartinreizabal/Documents/TFG/DatosEnVentanas/Benign_3.csv",   50),
    ("DoS-TCP",  "/Users/sergiomartinreizabal/Documents/TFG/DatosEnVentanas/DoS-TCP-0_00001_Flow_ventanas.csv",  50),
]

TOTAL_VENTANAS = 100      # <-- número total deseado
VENTANA        = timedelta(seconds=5)
SALIDA         = Path("400V_Benigno100_DoS-TCP100.csv")
COL_TS, COL_WIN = "Timestamp", "Ventana_Inicio"
# ---------------------------------------------------------------------

def cargar_csv(path):
    df = pd.read_csv(path)
    df[COL_TS]  = pd.to_datetime(df[COL_TS])
    df[COL_WIN] = pd.to_datetime(df[COL_WIN])
    df.sort_values(COL_WIN, inplace=True)
    return df

# Carga única y punteros
cache, ptr = {}, {}
for _, ruta, _ in SECUENCIA:
    if ruta not in cache:
        cache[ruta] = cargar_csv(ruta)
        ptr[ruta]   = 0

resultado, cursor = [], pd.Timestamp("2023-01-01 00:00:00")
ventanas_emitidas = 0
seq_idx = 0

while ventanas_emitidas < TOTAL_VENTANAS:
    etiqueta, ruta, bloq_n = SECUENCIA[seq_idx]
    seq_idx = (seq_idx + 1) % len(SECUENCIA)  # avanza circularmente

    libre = TOTAL_VENTANAS - ventanas_emitidas
    n = min(bloq_n, libre)         # recorta último bloque si hace falta

    # --- saca n ventanas consecutivas del CSV -------------------------
    df_full = cache[ruta]
    pos     = ptr[ruta]
    win_list = df_full[COL_WIN].drop_duplicates().values

    if pos + n > len(win_list):
        raise ValueError(
            f"No quedan {n} ventanas libres en '{ruta}'. "
            "Añade más CSVs o reduce TOTAL_VENTANAS."
        )

    sel_wins = win_list[pos : pos + n]
    bloque   = df_full[df_full[COL_WIN].isin(sel_wins)].copy()
    ptr[ruta] += n

    bloque["Label"] = etiqueta

    delta = cursor - bloque[COL_WIN].iloc[0]
    bloque[COL_TS]  += delta
    bloque[COL_WIN] += delta

    resultado.append(bloque)
    cursor += n * VENTANA
    ventanas_emitidas += n

# Concatenar y guardar
out = pd.concat(resultado, ignore_index=True).sort_values(COL_TS)
out.to_csv(SALIDA, index=False)
print(f"CSV '{SALIDA}' creado ({ventanas_emitidas} ventanas, "
      f"{len(out)} flujos, duración {(cursor - pd.Timestamp('2023-01-01'))}).")
