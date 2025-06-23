# app/modelo/utils.py
# ────────────────────────────────────────────────────────────────────────────
# 1) Partir flujos (CICFlowMeter) en ventanas fijas de 5 s
# 2) Convertir cada ventana en tensor [L, 76] usando EXACTAMENTE las columnas
#    numéricas del entrenamiento (train_cols)
# 3) Dataset PyTorch que aplica el StandardScaler ya entrenado
# ────────────────────────────────────────────────────────────────────────────
from datetime import timedelta
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ────────────────────────────── 1 · Ventanas ───────────────────────────────
def procesar_flujos_por_ventanas_from_df(
    df: pd.DataFrame,
    etiqueta_global: str = "__dummy__",    # se añade si el CSV no trae Label
    ventana_size: int = 5,
) -> pd.DataFrame:
    """Devuelve df con todos los flujos + columna Ventana_Inicio."""
    if {"Timestamp", "Flow Duration"}.difference(df.columns):
        raise ValueError("El CSV necesita columnas 'Timestamp' y 'Flow Duration'")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Label"] = etiqueta_global
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Timestamp", "Flow Duration"], inplace=True)

    ventanas: Dict[pd.Timestamp, List[pd.Series]] = {}
    eps = 1e-9

    for _, flujo in df.iterrows():
        ts_ini = flujo["Timestamp"]
        ts_fin = ts_ini + timedelta(microseconds=int(flujo["Flow Duration"]))
        ini_s, fin_s = ts_ini.timestamp(), ts_fin.timestamp()

        primera = int(ini_s // ventana_size) * ventana_size
        ultima  = int((fin_s - eps) // ventana_size) * ventana_size

        t = primera
        while t <= ultima:
            if ini_s < t + ventana_size and fin_s > t:
                ventanas.setdefault(pd.to_datetime(t, unit="s"), []).append(flujo)
            t += ventana_size

    filas = []
    for key, flujos in sorted(ventanas.items(), key=lambda kv: kv[0]):
        for f in flujos:
            d = f.to_dict()
            d["Ventana_Inicio"] = key
            filas.append(d)

    return pd.DataFrame(filas)


# ─────────────── 2 · Ventana → tensor con las columnas del training ────────
TRAIN_COLS: Optional[List[str]] = None   # se fija en la 1.ª llamada

def windows_to_tensors(
    df_windows: pd.DataFrame,
    max_seq_len: int = 127,
    train_cols: Optional[List[str]] = None,
    drop_cols: Sequence[str] = (
        "Ventana_Inicio", "Flow ID", "Src IP", "Src Port",
        "Dst IP", "Dst Port", "Protocol", "Timestamp", "Label",
    ),
) -> List[Tuple[np.ndarray, int]]:
    """
    Devuelve lista de (tensor, label_idx). Usa siempre el orden de `train_cols`.
    `train_cols` DEBE pasarse la primera vez (lista de 76 columnas numéricas).
    """
    global TRAIN_COLS
    if TRAIN_COLS is None:
        if train_cols is None:
            raise ValueError("train_cols no puede ser None la primera llamada")
        TRAIN_COLS = train_cols[:]        # congela lista exacta

    if df_windows.empty:
        return []

    if "Label" not in df_windows.columns:
        df_windows = df_windows.assign(Label="__dummy__")

    lbl2idx = {lbl: i for i, lbl in enumerate(sorted(df_windows["Label"].unique()))}
    samples: List[Tuple[np.ndarray, int]] = []

    for _, sub in df_windows.groupby("Ventana_Inicio"):
        sub = sub.sort_values("Timestamp")
        feats = sub[TRAIN_COLS].to_numpy(dtype=np.float32)

        if feats.shape[0] > max_seq_len:
            feats = feats[:max_seq_len]
        elif feats.shape[0] < max_seq_len:
            pad = np.zeros(
                (max_seq_len - feats.shape[0], feats.shape[1]), dtype=np.float32
            )
            feats = np.vstack([feats, pad])

        label_idx = lbl2idx[sub["Label"].iloc[0]]
        samples.append((feats, label_idx))

    return samples


# ─────────────────────────── 3 · Dataset PyTorch ───────────────────────────
class WindowDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]], scaler, train_cols):
        self.samples = samples
        self.scaler = scaler
        self.train_cols = train_cols

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        arr, lbl = self.samples[idx]                   # arr shape = (L, F)
        flat = arr.reshape(-1, arr.shape[1]).copy()    # (L, F) → (L·F, F)

        # 1) LIMPIAR ANTES DE ESCALAR  ----------------------------
        mask_bad = ~np.isfinite(flat)                  # NaN o ±Inf
        if mask_bad.any():
            # logueamos sólo la primera ocurrencia problemática
            r, c = np.argwhere(mask_bad)[0]
            col_name = self.train_cols[c] \
                       if c < len(self.train_cols) else f"col_{c}"
            raw_val = flat[r, c]
            print(f"Valor no finito en fila {r}, columna '{col_name}'. "
                  f"Raw={raw_val}  →  0.0")
            flat[mask_bad] = 0.0

        # 2) ESCALAR ---------------------------------------------
        norm = self.scaler.transform(flat).reshape(arr.shape)

        # 3) SEGURIDAD EXTRA (rara vez salta) ---------------------
        if not np.isfinite(norm).all():
            r, c = np.argwhere(~np.isfinite(norm))[0]
            col_name = self.train_cols[c] \
                       if c < len(self.train_cols) else f"col_{c}"
            raise ValueError(
                f"Scaler produjo valor no finito en fila {r}, "
                f"columna '{col_name}'")

        # 4) DEVOLVER TENSORES -----------------------------------
        return torch.from_numpy(norm), torch.tensor(lbl, dtype=torch.long)
