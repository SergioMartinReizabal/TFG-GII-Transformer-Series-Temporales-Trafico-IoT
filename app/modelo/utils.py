# -------------------------------------------------------------------------
# Funciones auxiliares para:
#   1) Partir flujos (CICFlowMeter CSV) en ventanas temporales de 5 s
#   2) Convertir esas ventanas en tensores fijos [L, 79]
#   3) Dataset PyTorch que aplica el StandarScaler ya entrenado
# -------------------------------------------------------------------------
import io
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def procesar_flujos_por_ventanas_from_df(
    df: pd.DataFrame,
    etiqueta_global: str,
    ventana_size: int = 5
) -> pd.DataFrame:
    """
    Agrupa un DataFrame de flujos individuales (salida nativa de CICFlowMeter)
    en ventanas fijas de 'ventana_size' segundos.
    
    - Añade/reescribe la columna 'Label' con 'etiqueta_global'.
    - Devuelve un nuevo DataFrame expandido: cada fila representa un flujo y
    la ventana a la que pertenece.
    
    Notas
    -----
    - Un mismo flujo puede aparecer en varias ventanas si dura >5 s.
    - Se ignoran valores 'inf' o NaN para evitar romper el modelo
    """
    
    if "Timestamp" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'Timestamp'")
    
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Label"] = etiqueta_global
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    ventanas: Dict[pd.Timestamp, List[pd.Series]] = {}
    eps = 1e9
    
    for _, flujo in df.iterrows():
        ts_ini = flujo["Timestamp"]
        ts_fin = ts_ini + timedelta(microseconds=int(flujo["Flow Duration"]))
        ini_s, fin_s = ts_ini.timestamp(), ts_fin.timestamp()
        
        primera = int(ini_s // ventana_size) * ventana_size
        ultima = int((fin_s -eps) // ventana_size) * ventana_size
        
        ventana_actual = primera
        while ventana_actual <= ultima:
            fin_ventana = ventana_actual + ventana_size
            if ini_s < fin_ventana and fin_s > ventana_actual:
                key = pd.to_datetime(ventana_actual, unit="s")
                ventanas.setdefault(key, []).append(flujo)
            ventana_actual += ventana_size
            
        filas = []
        for key, flujos in sorted(ventanas.items(), key=lambda kv: kv[0]):
            for f in flujos:
                d = f.to_dict()
                d["Ventana_Inicio"] = key
                filas.append(d)
        
        df_out = pd.DataFrame(filas)
        return df_out
    
    def windows_to_tensors(
        df_windows: pd.DataFrame,
        max_seq_len: int = 127,
        drop_cols: Sequence[str] = (
            "Ventana_Inicio", "Flow ID", "Src IP", "Src Port",
            "Dst IP", "Dst Port", "Protocol", "Timestamp", "Label"
        ) 
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Agrupa por 'Ventana_Inicio' y genera:
            - ndarray shape (L, n_features) en float32
            - entero con la etiqueta
        Devuelve la lista para enganchar con 'WindowDataset'
        """
        numeric_cols = df_windows.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).columns
        
        label_names = sorted(df_windows["Label"].unique())
        lbl2idx = {lbl: i for i, lbl in enumerate(label_names)}
        
        samples: List[Tuple[np.ndarray, int]] = []
        for _, sub in df_windows.groupby('Ventana_Inicio'):
            sub = sub.sort_values("Timestamp")
            label_idx = lbl2idx[sub["Label"].iloc[0]]
            feats = sub[numeric_cols].to_numpy(dtype=np.float32)
            
            if feats.shape[0] > max_seq_len:
                feats = feats[:max_seq_len]
            elif feats.shape[0] < max_seq_len:
                pad = np.zeros((max_seq_len - feats.shape[0], feats.shape[1]), dtype=np.float32)
                feats = np.vstack([feats, pad])
            
            samples.append((feats, label_idx))
        
        return samples