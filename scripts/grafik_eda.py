"""Membuat ulang grafik EDA utama dari data TRAIN (2025) dan menyimpannya ke docs/eda/.

Grafik ini sama dengan bagian 4 notebook ML_Kasus08_Final.ipynb, tetapi bisa dibuat tanpa
membuka notebook, misalnya untuk README dan laporan. Data validation dan test tidak dipakai.

Cara pakai (dari folder utama repositori):
    python scripts/grafik_eda.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')                                   # simpan ke file tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import hitung_volume_kemarin, rapikan  # noqa: E402

KELUARAN = ROOT / 'docs' / 'eda'


def muat_train():
    """Data unik, dirapikan, hanya Januari–Desember 2025 (train)."""
    df = pd.read_csv(ROOT / 'data' / 'raw.csv').drop_duplicates()
    df['tanggal_pickup'] = pd.to_datetime(df['tanggal_pickup'])
    volume_kemarin = hitung_volume_kemarin(df)          # fitur saja, dihitung dari volume hari sebelumnya
    train = rapikan(df[df['tanggal_pickup'] < '2026-01-01'].copy())
    train['volume_hub_kemarin'] = train['tanggal_pickup'].map(volume_kemarin)
    return train


def batang(ax, p, judul, rata):
    b = ax.bar(p.index.astype(str), p.values, color='tab:red', alpha=0.75)
    ax.bar_label(b, labels=[f'{v:.1f}%' for v in p.values], padding=3, fontsize=11)
    ax.axhline(rata, color='black', ls='--', lw=1.2, label=f'rata-rata {rata:.0f}%')
    ax.set_title(judul, fontweight='bold')
    ax.set_ylabel('% terlambat')
    ax.set_ylim(0, max(p.max() * 1.2, rata * 1.5))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)


def grafik_faktor(train, rata):
    b = train.copy()
    b['jarak'] = pd.cut(b['jarak_km'], [0, 200, 500, 1000, 2500], labels=['< 200', '200–500', '500–1000', '> 1000'])
    b['berat'] = pd.cut(b['berat_kg'], [0, 0.5, 1, 2, 5, 50], labels=['< 0,5', '0,5–1', '1–2', '2–5', '> 5'])
    b['jam'] = pd.cut(b['jam_pickup'], [6, 10, 14, 17, 22], labels=['07–10', '11–14', '15–17', '18–22'])
    b['cuaca'] = b['cuaca_saat_pickup'].fillna('tidak_diketahui')
    p = lambda kolom, urutan=None: (b.groupby(kolom, observed=True)['terlambat'].mean() * 100).reindex(urutan) \
        if urutan else b.groupby(kolom, observed=True)['terlambat'].mean() * 100

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    batang(ax[0, 0], p('jumlah_transit'), 'Jumlah transit', rata)
    batang(ax[0, 1], p('jarak'), 'Jarak (km)', rata)
    batang(ax[0, 2], p('layanan', ['EXPRESS', 'REG', 'HEMAT']), 'Layanan', rata)
    batang(ax[1, 0], p('cuaca', ['cerah', 'berawan', 'hujan', 'hujan_lebat', 'tidak_diketahui']), 'Cuaca saat pickup', rata)
    batang(ax[1, 1], p('jam'), 'Jam pickup', rata)
    batang(ax[1, 2], p('berat'), 'Berat (kg): tidak berpengaruh', rata)
    ax[1, 0].tick_params(axis='x', rotation=20)
    fig.suptitle('TRAIN 2025: proporsi paket terlambat menurut faktor', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(KELUARAN / 'faktor_keterlambatan.png', dpi=130)
    plt.close(fig)


def grafik_musim(train):
    harian = train.groupby('tanggal_pickup').agg(volume=('volume_hub_hari', 'median'), terlambat=('terlambat', 'mean'))
    bulanan = harian.resample('MS').mean()
    fig, ax1 = plt.subplots(figsize=(14, 5.5))
    ax1.bar(bulanan.index, bulanan['terlambat'] * 100, width=25, align='edge', color='tab:red', alpha=0.6, label='% terlambat')
    ax1.set_ylabel('% terlambat')
    ax2 = ax1.twinx()
    ax2.plot(bulanan.index + pd.Timedelta(days=12), bulanan['volume'], color='tab:blue', marker='o', lw=2.5, label='volume hub per hari')
    ax2.set_ylabel('volume hub per hari')
    ax1.set_title('TRAIN 2025: keterlambatan melonjak bersama volume hub (Maret, Nov–Des)', fontweight='bold')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.88))
    fig.tight_layout()
    fig.savefig(KELUARAN / 'musim_sibuk.png', dpi=130)
    plt.close(fig)


def grafik_kelonggaran(train):
    kelonggaran = train['janji_hari'] - train['jumlah_transit']
    p = train.groupby(kelonggaran)['terlambat'].agg(['mean', 'size'])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b = ax.bar(p.index.astype(str), p['mean'] * 100, color='tab:red', alpha=0.75)
    ax.bar_label(b, labels=[f'{v:.0f}%\n(n={n:,})' for v, n in zip(p['mean'] * 100, p['size'])], padding=4, fontsize=11)
    ax.set_xlabel('kelonggaran janji = janji_hari − jumlah_transit')
    ax.set_ylabel('% terlambat')
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('TRAIN 2025: makin mepet janjinya, makin sering terlambat', fontweight='bold')
    fig.tight_layout()
    fig.savefig(KELUARAN / 'kelonggaran_janji.png', dpi=130)
    plt.close(fig)


def main():
    KELUARAN.mkdir(parents=True, exist_ok=True)
    train = muat_train()
    rata = train['terlambat'].mean() * 100
    grafik_faktor(train, rata)
    grafik_musim(train)
    grafik_kelonggaran(train)
    print(f'Train: {len(train):,} paket, {rata:.1f}% terlambat. Grafik disimpan di {KELUARAN.relative_to(ROOT)}/')


if __name__ == '__main__':
    main()
