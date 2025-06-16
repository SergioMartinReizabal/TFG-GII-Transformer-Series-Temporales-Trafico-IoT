from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd, torch, joblib, json, io
from modelo.utils import procesar_flujos_por_ventanas_from_df, windows_to_tensors, WindowDataset
from modelo.arch import TransformerEncoderClassifierWithCLS

# -- carga de artefactos en el arranque ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map   = json.load(open("export/label_map.json"))
idx_to_lbl  = {v:k for k,v in label_map.items()}
scaler      = joblib.load("export/scaler.pkl")

model = TransformerEncoderClassifierWithCLS(
        d_model=64, num_heads=4, d_ff=128, num_layers=2,
        input_dim=79, num_classes=len(label_map), max_seq_len=128)
model.load_state_dict(torch.load("export/model.pth", map_location=device))
model.to(device).eval()

app = FastAPI(title="TFG IoT-Traffic API")

# ---------------------------- endpints --------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(csv: UploadFile = File(...), etiqueta: str = "UNKNOWN"):
    if not csv.filename.endswith(".csv"):
        raise HTTPException(400, "Solo se aceptan CSVs")
    content = await csv.read()
    df = pd.read_csv(io.BytesIO(content))
    
    # 1. agrupar a ventanas (reutiliza tu función original)
    df_windows = procesar_flujos_por_ventanas_from_df(df, etiqueta)
    
    # 2. crear tensor [L,79] por ventana
    samples = windows_to_tensors(df_windows, max_seq_len=127)  # ⇢ list[(tensor, _)]
    dataset = WindowDataset(samples, scaler)
    
    outs = []
    with torch.no_grad():
        for x, _ in dataset:
            logits = model(x.unsqueeze(0).to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            lbl    = idx_to_lbl[int(probs.argmax())]
            outs.append({"label": lbl,
                         "confidence": float(probs.max())})
    return JSONResponse(outs)
