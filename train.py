"""Melatih model final, mengikuti bagian 1–9 notebook `notebooks/ML_Kasus08_Final.ipynb`.

    python train.py

Hanya memakai TRAIN (2025) dan VALIDATION (Jan–Mar 2026). Data TEST (Apr–Jun 2026)
dipisahkan dan TIDAK disentuh di sini; evaluasi TEST hanya ada di bagian 10 notebook
dan dijalankan satu kali.
"""
import hashlib
import itertools
import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (ANGKA_FINAL, FITUR_ASLI_ANGKA, FITUR_KATEGORI, KOLOM_FINAL, TARGET,
                          buat_fitur, hitung_volume_kemarin, isi_kosong, rapikan)

ROOT = Path(__file__).resolve().parent
SHA_DATA = '2821b50eac1b02181cacbfb52fca784b2b5be930c73e5c671a877eeec6a600ab'
BIAYA_FN, BIAYA_FP = 50_000, 5_000
BATAS_TEST, BATAS_VAL = '2026-04-01', '2026-01-01'


def pisah_waktu(df):
    """Membuang duplikat persis lalu memisahkan kerja (train+val) dan test berdasarkan tanggal."""
    df = df.drop_duplicates().reset_index(drop=True)
    assert df['id_resi'].is_unique, 'Ada resi ganda dengan isi berbeda'
    df['tanggal_pickup'] = pd.to_datetime(df['tanggal_pickup'])
    test_mentah = df[df['tanggal_pickup'] >= BATAS_TEST].copy()
    kerja = df[df['tanggal_pickup'] < BATAS_TEST].copy()
    return df, kerja, test_mentah


def pipa(model, angka, skala=True):
    langkah_angka = [('isi', SimpleImputer(strategy='median'))] + ([('skala', StandardScaler())] if skala else [])
    olah = ColumnTransformer([
        ('angka', Pipeline(langkah_angka), angka),
        ('kategori', Pipeline([('isi', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), FITUR_KATEGORI),
    ])
    return Pipeline([('olah', olah), ('model', model)])


def hitung_biaya(y, proba, t):
    tn, fp, fn, tp = confusion_matrix(y, (proba >= t).astype(int), labels=[0, 1]).ravel()
    return {'threshold': float(t), 'dikabari': int(tp + fp), 'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
            'recall': float(tp / (tp + fn)), 'biaya (Rp)': int(fn * BIAYA_FN + fp * BIAYA_FP)}


def pilih_threshold(y, proba):
    """Threshold 0,01–0,90 (langkah 0,01) dengan total biaya validation paling kecil."""
    kurva = pd.DataFrame([hitung_biaya(y, proba, t) for t in np.round(np.arange(0.01, 0.91, 0.01), 2)])
    return float(kurva.loc[kurva['biaya (Rp)'].idxmin(), 'threshold'])


def latih(sumber=ROOT / 'data' / 'raw.csv', keluaran=ROOT / 'models'):
    isi = Path(sumber).read_bytes()
    assert hashlib.sha256(isi).hexdigest() == SHA_DATA, 'Dataset berbeda!'
    df, kerja, test_mentah = pisah_waktu(pd.read_csv(sumber))
    volume_kemarin = hitung_volume_kemarin(df)        # fitur saja (bukan target)

    kerja = rapikan(kerja)
    train = kerja[kerja['tanggal_pickup'] < BATAS_VAL].copy()
    val = kerja[kerja['tanggal_pickup'] >= BATAS_VAL].copy()
    patokan = {'jarak_per_kota': train.groupby('kota_tujuan')['jarak_km'].median(),
               'median_berat': train['berat_kg'].median()}      # HANYA dari train
    train_f = buat_fitur(isi_kosong(train, patokan), volume_kemarin)
    val_f = buat_fitur(isi_kosong(val, patokan), volume_kemarin)
    y_train, y_val = train_f[TARGET].values, val_f[TARGET].values
    bulan_val = val_f['tanggal_pickup'].dt.month.values

    def nilai_ap(proba):
        return {'PR-AUC val': float(average_precision_score(y_val, proba)),
                'PR-AUC Jan+Feb': float(average_precision_score(y_val[bulan_val < 3], proba[bulan_val < 3]))}

    # Baseline wajib: regresi logistik, fitur asli, tanpa volume
    fitur_baseline = FITUR_ASLI_ANGKA + FITUR_KATEGORI
    baseline = pipa(LogisticRegression(max_iter=3000), FITUR_ASLI_ANGKA).fit(train_f[fitur_baseline], y_train)
    p_base = baseline.predict_proba(val_f[fitur_baseline])[:, 1]

    # Kandidat model final (sama dengan tabel tuning di bagian 8 notebook)
    kandidat = []
    for C in [0.1, 1, 10]:
        kandidat.append(('Regresi logistik', f'C={C}', pipa(LogisticRegression(C=C, max_iter=3000), ANGKA_FINAL)))
    for lr, depth, leaf in itertools.product([0.03, 0.1], [3, None], [20, 100]):
        kandidat.append(('Gradient Boosting', f'lr={lr}, depth={depth}, leaf={leaf}',
                         pipa(HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, min_samples_leaf=leaf,
                                                             max_iter=200, random_state=42), ANGKA_FINAL, skala=False)))
    hasil = []
    for nama, atur, m in kandidat:
        m.fit(train_f[KOLOM_FINAL], y_train)
        hasil.append({'model': nama, 'pengaturan': atur, 'objek': m, **nilai_ap(m.predict_proba(val_f[KOLOM_FINAL])[:, 1])})
    tabel = pd.DataFrame(hasil).sort_values('PR-AUC val', ascending=False, kind='stable').reset_index(drop=True)
    terbaik = tabel.iloc[0]
    model_final = terbaik['objek']
    nama_model = f"{terbaik['model']} ({terbaik['pengaturan']})"
    proba_final = model_final.predict_proba(val_f[KOLOM_FINAL])[:, 1]

    threshold = pilih_threshold(y_val, proba_final)
    threshold_base = pilih_threshold(y_val, p_base)

    keluaran = Path(keluaran)
    keluaran.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_final, keluaran / 'model.joblib')
    metadata = {
        'nama_model': nama_model, 'tanggal_dilatih': str(date.today()),
        'versi': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                  'pandas': pd.__version__, 'numpy': np.__version__},
        'sumber_data': 'https://github.com/nurhikam/mkb-ml-project/tree/main/kasus/08-keterlambatan-paket',
        'sha256_data': SHA_DATA,
        'baris': {'mentah': int(len(isi.splitlines()) - 1), 'unik': int(len(df)), 'train': int(len(train)),
                  'validation': int(len(val)), 'test_disimpan_tidak_dievaluasi': int(len(test_mentah))},
        'periode_train': '2025-01-01 s.d. 2025-12-31', 'periode_validation': '2026-01-01 s.d. 2026-03-31',
        'fitur_angka': ANGKA_FINAL, 'fitur_kategori': FITUR_KATEGORI,
        'threshold_kabari': threshold, 'threshold_prioritaskan': 0.50,
        'biaya': {'terlewat_FN': BIAYA_FN, 'salah_kabar_FP': BIAYA_FP},
        'patokan_cleaning': {'median_berat': float(patokan['median_berat']),
                             'jarak_per_kota': {k: float(v) for k, v in patokan['jarak_per_kota'].items()}},
        'metrik_validation': {'pr_auc_model': round(float(average_precision_score(y_val, proba_final)), 3),
                              'pr_auc_baseline': round(float(average_precision_score(y_val, p_base)), 3),
                              'pr_auc_acak': round(float(y_val.mean()), 3),
                              'pr_auc_jan_feb_model': round(nilai_ap(proba_final)['PR-AUC Jan+Feb'], 3),
                              'pr_auc_jan_feb_baseline': round(nilai_ap(p_base)['PR-AUC Jan+Feb'], 3),
                              'biaya_model_rp': hitung_biaya(y_val, proba_final, threshold)['biaya (Rp)'],
                              'biaya_baseline_rp': hitung_biaya(y_val, p_base, threshold_base)['biaya (Rp)'],
                              'biaya_kabari_semua_rp': hitung_biaya(y_val, np.ones(len(y_val)), 0.5)['biaya (Rp)'],
                              'konfusi_model': hitung_biaya(y_val, proba_final, threshold)},
        'tabel_tuning': tabel.drop(columns='objek').round(4).to_dict('records'),
        'evaluasi_test': 'belum dijalankan oleh train.py; lihat bagian 10 notebook (sekali saja)',
    }
    (keluaran / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    m = latih()
    print(json.dumps(m['metrik_validation'], indent=2, ensure_ascii=False))
    print('Model:', m['nama_model'], '| threshold:', m['threshold_kabari'])
