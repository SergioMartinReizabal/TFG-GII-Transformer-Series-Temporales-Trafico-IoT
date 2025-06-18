# app/main.py ---------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd, torch, joblib, json, io, sys, math, asyncio
from pathlib import Path
from pandas import Timestamp, Timedelta

# --- importa tu código local
sys.path.append(str(Path(__file__).parent))          # añade app/ al path
from modelo.utils import procesar_flujos_por_ventanas_from_df, windows_to_tensors, WindowDataset
from modelo.arch  import TransformerEncoderClassifierWithCLS

# ------------------ carga artefactos una sola vez -------------------
EXPORT = Path(__file__).parent / "modelo" / "export"
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map   = json.load(open(EXPORT/"label_map.json"))
idx_to_lbl  = {v:k for k,v in label_map.items()}
scaler      = joblib.load(EXPORT/"scaler.pkl")

with open(EXPORT / "numeric_cols.json") as fp:
    TRAIN_COLS = json.load(fp)        # orden original de 76 columnas

model = TransformerEncoderClassifierWithCLS(
    d_model=64, num_heads=4, d_ff=128, num_layers=2,
    input_dim=76, num_classes=len(label_map), max_seq_len=128
).to(device)
model.load_state_dict(torch.load(EXPORT/"model2.pth", map_location=device))
model.eval()

# ----------------------------- FastAPI --------------------------------
app = FastAPI(title="TFG · Clasificador IoT")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(csv: UploadFile = File(...), etiqueta: str = "UNKNOWN"):
    if not csv.filename.endswith(".csv"):
        raise HTTPException(400, detail="Solo se aceptan archivos .csv")
    content = await csv.read()
    df = pd.read_csv(io.BytesIO(content))

    df_windows = procesar_flujos_por_ventanas_from_df(df, etiqueta_global=etiqueta)
    samples    = windows_to_tensors(df_windows, max_seq_len=127, train_cols=TRAIN_COLS )
    dataset    = WindowDataset(samples, scaler)
    
    # Obtener los incios de ventana en el mismo orden que 'samples'
    starts = sorted(df_windows["Ventana_Inicio"].unique())

    outs = []
    with torch.no_grad():
        for (x, _), start in zip(dataset, starts):
            ts_start = Timestamp(start)
            logits = model(x.unsqueeze(0).to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            conf   = float(probs.max())
            intervalo = f"{ts_start} - {start + Timedelta(seconds=5)}"
            outs.append({
                "interval": intervalo,
                "label": idx_to_lbl[int(probs.argmax())],
                "confidence": conf
            })
    return JSONResponse(outs, status_code=200)

# sirve el HTML desde / (http://localhost:10000/)
app.mount("/", StaticFiles(directory=Path(__file__).parents[1]/"frontend", html=True), name="static")
