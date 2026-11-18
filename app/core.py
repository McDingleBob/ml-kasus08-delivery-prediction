"""Skema input dan logika prediksi. Terpisah dari FastAPI agar bisa diuji langsung."""
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.features import KOLOM_FINAL, buat_fitur, isi_kosong, rapikan

ROOT = Path(__file__).resolve().parents[1]

KOTA = ['Balikpapan', 'Bekasi', 'Bogor', 'Cirebon', 'Denpasar', 'Jakarta', 'Makassar', 'Malang',
        'Medan', 'Palembang', 'Pontianak', 'Semarang', 'Surabaya', 'Tasikmalaya', 'Yogyakarta']


class Paket(BaseModel):
    """Data satu paket saat dijemput. Kolom bertanda opsional boleh dikosongkan (null)."""
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    jam_pickup: int = Field(ge=7, le=22, description='Jam paket dijemput (7–22).')
    kota_tujuan: Literal[tuple(KOTA)]
    jarak_km: float | None = Field(default=None, gt=0, le=3000,
                                   description='Opsional. Jika kosong diisi median jarak kota yang sama di data train.')
    jumlah_transit: int = Field(ge=1, le=10, description='Jumlah hub transit (data latih: 1–5).')
    layanan: Literal['EXPRESS', 'REG', 'HEMAT']
    janji_hari: int = Field(ge=1, le=14, description='Janji waktu tiba (SLA) dalam hari.')
    berat_kg: float | None = Field(default=None, gt=0, le=50,
                                   description='Opsional, dalam kg (maks. 50). Paket terberat di data latih ±44 kg.')
    metode_bayar: Literal['COD', 'non_COD']
    cuaca_saat_pickup: Literal['cerah', 'berawan', 'hujan', 'hujan_lebat', 'tidak_diketahui'] | None = None
    volume_hub_kemarin: float | None = Field(default=None, ge=0, le=50000,
                                           description='Opsional. Jumlah paket yang masuk hub KEMARIN. '
                                                       'Jika kosong diisi median data train.')


class Prediktor:
    def __init__(self, folder=ROOT / 'models'):
        self.meta = json.loads((Path(folder) / 'metadata.json').read_text(encoding='utf-8'))
        self.model = joblib.load(Path(folder) / 'model.joblib')
        self.patokan = self.meta['patokan_cleaning']

    def prediksi(self, paket: Paket) -> dict:
        data = paket.model_dump()
        kosong = [k for k, v in data.items() if v is None]
        df = pd.DataFrame([data])
        for kolom in ['jarak_km', 'berat_kg', 'volume_hub_kemarin']:
            df[kolom] = pd.to_numeric(df[kolom]).astype(float)
        df = buat_fitur(isi_kosong(rapikan(df), self.patokan))
        p = float(self.model.predict_proba(df[KOLOM_FINAL])[0, 1])

        t_kabari, t_prior = self.meta['threshold_kabari'], self.meta['threshold_prioritaskan']
        if p >= t_prior:
            rekomendasi, saran = 'PRIORITASKAN', 'Risiko tinggi: prioritaskan paket ini dan kabari pelanggan.'
        elif p >= t_kabari:
            rekomendasi, saran = 'KABARI PELANGGAN', 'Ada risiko terlambat: kabari pelanggan lebih awal.'
        else:
            rekomendasi, saran = 'AMAN', 'Risiko rendah. Bukan jaminan paket tepat waktu.'
        return {'probabilitas_terlambat': round(p, 4), 'rekomendasi': rekomendasi, 'saran': saran,
                'threshold_kabari': t_kabari, 'threshold_prioritaskan': t_prior,
                'diisi_otomatis': kosong, 'model': self.meta['nama_model'],
                'catatan': 'Demo pembelajaran dengan data sintetis.'}
