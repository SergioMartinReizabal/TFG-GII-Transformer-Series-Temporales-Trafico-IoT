# app/main.py ----------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import io, json, math, asyncio, sys
from pathlib import Path
from datetime import timedelta

import pandas as pd, torch, joblib
import numpy as np
from pandas import Timestamp, Timedelta
from sklearn.metrics import confusion_matrix, classification_report

# --- añade app/ al sys.path y carga utilidades ------------------------
sys.path.append(str(Path(__file__).parent))
from modelo.utils import (
        procesar_flujos_por_ventanas_from_df,
        windows_to_tensors,
        WindowDataset)

from modelo.arch  import TransformerEncoderClassifierWithCLS

# -------- artefactos --------------------------------------------------
EXP      = Path(__file__).parent / "modelo" / "export"
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_map  = json.load(open(EXP/'label_map.json'))
idx2lbl    = {v:k for k,v in label_map.items()}
scaler     = joblib.load(EXP/'scaler.pkl')
NUM_COLS   = json.load(open(EXP/'numeric_cols.json'))   # 76 columnas

model = TransformerEncoderClassifierWithCLS(
            d_model=64, num_heads=4, d_ff=128, num_layers=2,
            input_dim=len(NUM_COLS), num_classes=len(label_map),
            max_seq_len=128).to(device)
model.load_state_dict(torch.load(EXP/'model2.pth', map_location=device))
model.eval()

# -------------------- FASTAPI -----------------------------------------
app = FastAPI(title="TFG · Clasificador IoT")

@app.post("/predict")
async def predict(csv: UploadFile = File(...),
                  etiqueta: str = "UNKNOWN"):
    if not csv.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Solo se aceptan archivos .csv")

    df_raw = pd.read_csv(io.BytesIO(await csv.read()))
    has_label = "Label" in df_raw.columns

    # --- decidir si ya existe la ventana -----------------------------
    if "Ventana_Inicio" in df_raw.columns:
        #  ➜  caso B  (ya vienen asignadas)
        df_windows = df_raw
    else:
        #  ➜  caso A  (hay que agrupar en ventanas)
        df_windows = procesar_flujos_por_ventanas_from_df(
                         df_raw.copy(), etiqueta_global=etiqueta)

    # --- construir tensores -----------------------------------------
    df_windows["Ventana_Inicio"] = pd.to_datetime(
            df_windows["Ventana_Inicio"], errors='coerce')

    samples  = windows_to_tensors(df_windows,
                                max_seq_len=127,
                                train_cols=NUM_COLS)
    
    dataset  = WindowDataset(samples, scaler, NUM_COLS)

    starts = sorted(df_windows["Ventana_Inicio"].unique())
    assert len(starts) == len(dataset), "desfase ventanas ↔ tensores"

    # ordenar los inicios para ir sincronizados
    starts = sorted(df_windows["Ventana_Inicio"].unique())
    starts = [pd.Timestamp(t) if isinstance(t, np.datetime64) else t for t in starts]
    assert len(starts) == len(dataset), "desfase ventanas ↔ tensores"

    # --- etiquetas reales por ventana (si las trae) -----------------
    y_true = []
    if has_label:
        # df_raw.Timstamp → datetime
        if df_raw["Timestamp"].dtype.kind != 'M':
            df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"],
                                                 errors='coerce')
        for s in starts:
            e = s + Timedelta(seconds=5)
            mask  = (df_raw["Timestamp"]>=s) & (df_raw["Timestamp"]<e)
            lbls  = df_raw.loc[mask, "Label"]
            y_true.append(lbls.mode().iloc[0] if not lbls.empty else "__void__")

    # --- inferencia ---------------------------------------------------
    preds, y_pred = [], []
    with torch.no_grad():
        for (x,_), s in zip(dataset, starts):
            logits = model(x.unsqueeze(0).to(device))
            probs  = torch.softmax(logits,1).cpu().numpy()[0]
            cls    = int(probs.argmax())
            y_pred.append(idx2lbl[cls])
            
            ini = s.strftime("%Y-%m-%d %H:%M:%S")
            fin = (s + pd.Timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S")

            preds.append({
                "interval"  : f"{ini} - {fin}",
                "label"     : idx2lbl[cls],
                "confidence": float(probs.max()),
                **({"true": y_true[len(y_pred)-1]} if has_label else {})
            })

    # --- métricas -----------------------------------------------------
    evaluation = None
    if has_label:
        labels_eval = sorted(set(y_true + y_pred) - {"__void__"})
        cm = confusion_matrix(y_true, y_pred, labels=labels_eval)
        rep = classification_report(
                y_true, y_pred, labels=labels_eval,
                output_dict=True, zero_division=0)

        evaluation = {
            "labels"   : labels_eval,
            "confusion": cm.tolist(),
            "metrics"  : {
                "accuracy"   : rep["accuracy"],
                "f1_weight": rep["weighted avg"]["f1-score"]
            }
        }

    return JSONResponse({"pred": preds, "evaluation": evaluation})


# ---------- servir frontend -------------------------------------------
app.mount("/", StaticFiles(directory=Path(__file__).parents[1]/"frontend",
                           html=True), name="static")
