"""Cleaning dan fitur, sama persis dengan bagian 3.2 dan 5 notebook `ML_Kasus08_Final.ipynb`.

Dipakai oleh train.py (pelatihan) dan app/ (aplikasi), sehingga data latih dan input
aplikasi diolah dengan cara yang sama.
"""
import numpy as np
import pandas as pd

PETA_KOTA = {'jakarta': 'Jakarta', 'jkt': 'Jakarta', 'dki jakarta': 'Jakarta',
             'bekasi': 'Bekasi', 'kota bekasi': 'Bekasi', 'surabaya': 'Surabaya', 'sby': 'Surabaya',
             'medan': 'Medan', 'yogyakarta': 'Yogyakarta', 'di yogyakarta': 'Yogyakarta',
             'jogja': 'Yogyakarta', 'yogya': 'Yogyakarta', 'denpasar': 'Denpasar', 'bali': 'Denpasar'}

FITUR_ASLI_ANGKA = ['jam_pickup', 'jarak_km', 'jumlah_transit', 'janji_hari', 'berat_kg']
FITUR_KATEGORI = ['kota_tujuan', 'layanan', 'metode_bayar', 'cuaca_saat_pickup']
FITUR_BUATAN = ['volume_hub_kemarin', 'volume_kemarin_tinggi', 'kelonggaran_janji', 'pickup_malam']
ANGKA_FINAL = FITUR_ASLI_ANGKA + FITUR_BUATAN
KOLOM_FINAL = ANGKA_FINAL + FITUR_KATEGORI
TARGET = 'terlambat'


def rapikan(data):
    """Perbaikan yang tidak butuh statistik apa pun: ejaan, kode error, salah satuan."""
    d = data.copy()
    kota = d['kota_tujuan'].str.strip().str.lower()
    d['kota_tujuan'] = kota.map(PETA_KOTA).fillna(kota.str.title())
    d['layanan'] = d['layanan'].str.strip().str.upper().replace({'EXP': 'EXPRESS'})
    d.loc[d['berat_kg'] == -1, 'berat_kg'] = np.nan                       # kode tidak tercatat
    gram = d['berat_kg'] > 50
    d.loc[gram, 'berat_kg'] = d.loc[gram, 'berat_kg'] / 1000              # gram -> kg
    return d


def isi_kosong(data, patokan):
    """Mengisi nilai kosong memakai patokan yang dihitung dari TRAIN."""
    d = data.copy()
    d['jarak_km'] = d['jarak_km'].fillna(d['kota_tujuan'].map(patokan['jarak_per_kota']))
    d['berat_kg'] = d['berat_kg'].fillna(patokan['median_berat'])
    d['cuaca_saat_pickup'] = d['cuaca_saat_pickup'].fillna('tidak_diketahui')
    return d


def hitung_volume_kemarin(df):
    """Volume hub hari sebelumnya per tanggal (median volume_hub_hari per hari, digeser 1 hari)."""
    tanggal = pd.to_datetime(df['tanggal_pickup'])
    return df.groupby(tanggal)['volume_hub_hari'].median().shift(1)


def buat_fitur(data, volume_kemarin=None):
    f = data.copy()
    if 'volume_hub_kemarin' not in f:                                  # di aplikasi, nilai ini diisi petugas
        f['volume_hub_kemarin'] = f['tanggal_pickup'].map(volume_kemarin)
    f['kelonggaran_janji'] = f['janji_hari'] - f['jumlah_transit']
    f['pickup_malam'] = (f['jam_pickup'] >= 18).astype(int)
    f['volume_kemarin_tinggi'] = (f['volume_hub_kemarin'] > 9000).astype(int)
    return f
