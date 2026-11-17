"""Pickup-time risk API. All learned transforms live inside the model pipeline."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import json,logging,time
import joblib,pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel,Field,ConfigDict
ROOT=Path(__file__).resolve().parents[2]
state={}

@asynccontextmanager
async def lifespan(app):
    state['meta']=json.loads((ROOT/'artifacts/metadata.json').read_text(encoding='utf-8'))
    state['model']=joblib.load(ROOT/'artifacts/model.joblib')
    yield
    state.clear()

app=FastAPI(title='Paket Tepat Waktu · Bob & Riyu · DRAFT',version='0.1.0',lifespan=lifespan,description='Kasus08, data sintetis. Masukkan data saat pickup. Belum tervalidasi untuk operasi nyata. Kolom null diimputasi menggunakan median/modus TRAIN, bukan data baru.')

class Paket(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    jam_pickup:int=Field(ge=7,le=22)
    kota_tujuan:Literal['Tasikmalaya','Surabaya','Bekasi','Pontianak','Denpasar','Cirebon','Yogyakarta','Makassar','Bogor','Jakarta','Semarang','Palembang','Medan','Malang','Balikpapan']
    jarak_km:float|None=Field(default=None,gt=0,le=3000)
    jumlah_transit:int=Field(ge=0,le=10,description='Asumsi belum terkonfirmasi: jumlah transit terencana tersedia saat pickup. Brief tidak memastikan planned versus realized; konfirmasi schema upstream masih terbuka. Jika retrospektif, model tidak layak untuk deployment pickup-time; perlu eksperimen prospektif dan rencana holdout baru, bukan tuning TEST lama.')
    layanan:Literal['EXPRESS','REG','HEMAT']
    janji_hari:int=Field(ge=1,le=14)
    berat_kg:float|None=Field(default=None,gt=0,le=100,description='Satuan kg. Batas 100 kg hanya lingkup demo, bukan batas industri/brief atau bukti salah satuan. API menolak angka <=0 atau >100 dengan 422; training memetakan angka tersebut menjadi missing lalu median TRAIN. Null diizinkan dan diimputasi pipeline frozen; tidak ada konversi gram.')
    metode_bayar:Literal['COD','non_COD']
    cuaca_saat_pickup:Literal['cerah','berawan','hujan','hujan_lebat']|None=None

@app.get('/',include_in_schema=False)
def home(): return RedirectResponse('/docs')

@app.get('/health')
def health():
    m=state['meta']
    return {'status':'ok','case':m['case'],'model':m['selected_model'],'version':m['version'],'threshold':m['threshold'],'features':m['features'],'synthetic_data':True}

@app.post('/predict')
def predict(paket:Paket):
    start=time.perf_counter(); m=state['meta']
    p=float(state['model'].predict_proba(pd.DataFrame([paket.model_dump()]))[0,1])
    recommendation='KABARI PELANGGAN DAN PRIORITASKAN' if p>=m['threshold'] else 'AMAN'
    result={'probabilitas_terlambat':p,'rekomendasi':recommendation,'threshold':m['threshold'],'model_version':m['version'],'diimputasi':[k for k,v in paket.model_dump().items() if v is None],'latency_ms':round((time.perf_counter()-start)*1000,2),'batasan':'Demo data sintetis; AMAN bukan jaminan paket tepat waktu.'}
    logging.getLogger('uvicorn.error').info('prediction model=%s probability=%.6f recommendation=%s',m['version'],p,recommendation)
    return result
