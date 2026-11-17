"""Deterministic, non-learned cleaning shared between training and serving.

Full-day volume_hub_hari is excluded: at pickup its final value is unknown.
Nonpositive weight or weight >100kg is mapped to missing, then imputed
using the downstream TRAIN-fitted median. The 100kg cap is demo scope only,
not evidence of corrupt units or an industry/source limit; no gram conversion.
The API deliberately rejects these numeric weights (422), but allows null.
Planned jumlah_transit available at pickup is an unconfirmed assumption;
upstream planned-versus-realized provenance must be confirmed before use.
"""
import numpy as np
import pandas as pd
NUMERIC=['jam_pickup','jarak_km','jumlah_transit','janji_hari','berat_kg']
CATEGORICAL=['kota_tujuan','layanan','metode_bayar','cuaca_saat_pickup']
FEATURES=NUMERIC+CATEGORICAL
CITY={'sby':'Surabaya','surabaya':'Surabaya','dki jakarta':'Jakarta','jakarta':'Jakarta','jkt':'Jakarta','kota bekasi':'Bekasi','bekasi':'Bekasi','di yogyakarta':'Yogyakarta','yogya':'Yogyakarta','jogja':'Yogyakarta','yogyakarta':'Yogyakarta','bali':'Denpasar','denpasar':'Denpasar','medan':'Medan'}

def prepare(frame):
    out=frame.reindex(columns=FEATURES).copy()
    for c in NUMERIC:
        out[c]=pd.to_numeric(out[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    for c in CATEGORICAL:
        out[c]=out[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    out['kota_tujuan']=out['kota_tujuan'].map(lambda v:CITY.get(v.lower(),v.title()) if isinstance(v,str) else np.nan)
    out['layanan']=out['layanan'].map(lambda v:{'EXP':'EXPRESS','EXPRESS':'EXPRESS','REG':'REG','HEMAT':'HEMAT'}.get(v.upper(),np.nan) if isinstance(v,str) else np.nan)
    out.loc[(out.berat_kg<=0)|(out.berat_kg>100),'berat_kg']=np.nan
    out.loc[out.jarak_km<=0,'jarak_km']=np.nan
    return out
