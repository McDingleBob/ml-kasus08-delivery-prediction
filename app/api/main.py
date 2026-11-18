"""API prediksi keterlambatan paket.

    uvicorn app.api.main:app --host 0.0.0.0 --port 7860

GET /        halaman form (bisa dibuka dari ponsel)
GET /docs    dokumentasi interaktif (Swagger)
GET /health  status dan info model
POST /predict
"""
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.core import KOTA, Paket, Prediktor

STATIC = Path(__file__).resolve().parents[1] / 'static'


@lru_cache(maxsize=1)
def prediktor() -> Prediktor:
    """Model dimuat sekali saat pertama dibutuhkan (cocok untuk server biasa maupun serverless)."""
    return Prediktor()


app = FastAPI(title='Prediksi Keterlambatan Paket · Kasus 08', version='1.0.0',
              description='Bob & Riyu (MKB 5B). Masukkan data paket saat pickup; keluaran berupa probabilitas '
                          'terlambat dan rekomendasi. Data sintetis, bukan sistem operasional.')


@app.get('/', include_in_schema=False)
def halaman():
    return FileResponse(STATIC / 'index.html')


@app.get('/health')
def health():
    m = prediktor().meta
    return {'status': 'ok', 'model': m['nama_model'], 'threshold_kabari': m['threshold_kabari'],
            'threshold_prioritaskan': m['threshold_prioritaskan'], 'fitur_angka': m['fitur_angka'],
            'fitur_kategori': m['fitur_kategori'], 'kota': KOTA, 'data_sintetis': True}


@app.post('/predict')
def predict(paket: Paket):
    return prediktor().prediksi(paket)
