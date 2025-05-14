from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load model
with open("random_forest_menu_laris.pkl", "rb") as f:
    model = pickle.load(f)

# Inisialisasi aplikasi
app = FastAPI(title="Prediksi Menu Laris - Warung Makan")

# Skema input data
class MenuData(BaseModel):
    NamaMenu: int
    Kategori: int
    HargaSatuan: float
    TotalPenjualan: float
    Bulan: int
    KategoriHarga: int

# Endpoint untuk prediksi
@app.post("/predict")
def predict_menu(data: MenuData):
    # Konversi input ke DataFrame
    df = pd.DataFrame([data.dict()])

    # Prediksi
    prediction = model.predict(df)

    # Kembalikan hasil
    return {"prediksi_menu_laris": int(prediction[0])}

# Jalankan dengan: uvicorn fastapi_menu_laris:app --reload
