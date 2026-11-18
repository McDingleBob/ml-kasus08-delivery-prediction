# Prediksi Keterlambatan Paket — Kasus 08

Machine Learning · Kelas MKB 5B · **Bob Nolen Barnabas Munthe (2406157)** & **Muhammad Riyu Saputra (2403287)**

Saat paket dijemput di hub Bandung, model memperkirakan peluang paket terlambat (melewati `janji_hari`) dan memberi rekomendasi untuk staf layanan pelanggan dan operasional: **aman**, **kabari pelanggan**, atau **prioritaskan**.
Brief dan data: [nurhikam/mkb-ml-project · kasus 08](https://github.com/nurhikam/mkb-ml-project/tree/main/kasus/08-keterlambatan-paket). Data sintetis, jadi ini bahan pembelajaran, bukan sistem operasional.

| Tautan | |
|---|---|
| Aplikasi | _belum ada (isi setelah deploy)_ |
| Notebook Colab | _isi tautan Colab_ |
| Video | _isi tautan video_ |

## Isi repositori

| Path | Isi |
|---|---|
| `notebooks/ML_Kasus08_Final.ipynb` | Alur lengkap: cleaning, EDA, fitur, baseline, perbandingan model, threshold, evaluasi test |
| `src/features.py` | Fungsi cleaning dan fitur, sama persis dengan notebook; dipakai pelatihan dan aplikasi |
| `train.py` | Melatih ulang model final dari `data/raw.csv` (bagian 1–9 notebook, tanpa test) |
| `models/model.joblib`, `models/metadata.json` | Model final, threshold, patokan cleaning dari train, metrik validation |
| `models/hasil_test.json` | Hasil evaluasi test (satu kali) |
| `app/` | API FastAPI + halaman form yang bisa dibuka dari ponsel |
| `tests/` | Unit test cleaning, fitur, pelatihan, dan validasi input |
| `data/raw.csv` | Data asli dari pengajar (SHA-256 `2821b50e…a600ab`) |

## Ringkasan metode

- **Data:** 40.500 baris → 40.000 setelah membuang 500 duplikat persis. Tidak ada baris lain yang dibuang.
- **Split waktu:** train Jan–Des 2025 (26.633), validation Jan–Mar 2026 (6.623), test Apr–Jun 2026 (6.744, dipisahkan sejak awal).
- **Cleaning:** ejaan kota (28 varian → 15) dan layanan (7 → 3) disatukan; berat −1 → kosong; berat > 50 dibagi 1.000 (diduga gram); jarak kosong → median kota yang sama di train; berat kosong → median train; cuaca kosong → `tidak_diketahui`.
- **Fitur:** 9 fitur asli (tanpa `volume_hub_hari`), ditambah `volume_hub_kemarin`, `volume_kemarin_tinggi`, `kelonggaran_janji` (= janji − transit), dan `pickup_malam`. `volume_hub_hari` tidak dipakai karena total hari itu belum diketahui saat pickup.
- **Baseline wajib:** regresi logistik dengan fitur asli.
- **Model final:** regresi logistik (C = 0,1) dengan fitur buatan, dipilih dengan PR-AUC validation tertinggi.
- **Threshold:** 0,10, dipilih dengan biaya validation terkecil (paket terlambat tanpa kabar Rp50.000, kabar keliru Rp5.000). Probabilitas ≥ 0,50 → prioritaskan (kebijakan tim).

## Hasil validation (Jan–Mar 2026)

| | PR-AUC | PR-AUC Jan+Feb | Biaya |
|---|---:|---:|---:|
| Tebakan acak | 0,219 | – | – |
| Kabari semua paket | – | – | Rp25.870.000 |
| Baseline (LR fitur asli) | 0,409 | 0,371 | Rp25.870.000 |
| **Model final** | **0,605** | 0,373 | **Rp20.730.000** |

Model final mengabari 60% paket dan menangkap 90% paket yang terlambat. Kenaikan PR-AUC hampir seluruhnya berasal dari musim sibuk Maret 2026; di bulan biasa model hampir setara baseline.

## Hasil test (Apr–Jun 2026, dijalankan satu kali)

| | PR-AUC | Dikabari | Recall | Biaya |
|---|---:|---:|---:|---:|
| Tebakan acak | 0,139 | – | – | – |
| Tidak pernah mengabari | – | 0% | 0% | Rp46.950.000 |
| Kabari semua paket | – | 100% | 100% | Rp29.025.000 |
| Baseline (LR fitur asli, threshold 0,01) | 0,360 | 100% | 100% | Rp29.025.000 |
| **Model final (threshold 0,10)** | **0,362** | 53% | 80% | **Rp23.425.000** |

Di test (tanpa musim sibuk, keterlambatan 13,9%), PR-AUC model final hampir sama dengan baseline. Ini sejalan dengan hasil Jan+Feb di validation. Manfaat yang tersisa ada pada keputusan: biaya ±19% lebih rendah daripada mengabari semua paket. Model dan threshold tidak diubah setelah melihat hasil ini. Rincian ada di `models/hasil_test.json` dan bagian 10 notebook.

## Menjalankan

```sh
pip install -r requirements.txt
python -m unittest discover -s tests -v       # test
python train.py                               # melatih ulang (hasil sama dengan notebook)
uvicorn app.api.main:app --host 0.0.0.0 --port 7860
```

Buka `http://localhost:7860` untuk form, `/docs` untuk dokumentasi API, `/health` untuk status. Docker: `docker build -t kasus08 . && docker run -p 7860:7860 kasus08`.

Contoh `POST /predict`:
```json
{"jam_pickup": 20, "kota_tujuan": "Makassar", "jarak_km": 1500, "jumlah_transit": 4, "layanan": "EXPRESS",
 "janji_hari": 2, "berat_kg": 3, "metode_bayar": "COD", "cuaca_saat_pickup": "hujan_lebat", "volume_hub_kemarin": 11000}
```
`jarak_km`, `berat_kg`, `cuaca_saat_pickup`, dan `volume_hub_kemarin` boleh `null`; nilainya diisi dari patokan train. Input tidak wajar (jam 25, berat −1, kota tidak dikenal, kolom tambahan, dsb.) ditolak dengan HTTP 422 tanpa membuat aplikasi crash.

## Keterbatasan

- Data sintetis; biaya Rp0 untuk prediksi benar adalah asumsi.
- Belum dipastikan apakah `jumlah_transit` adalah rencana (tersedia saat pickup) atau realisasi.
- Konversi berat > 50 sebagai gram adalah dugaan, bukan kepastian dari klien.
- `volume_hub_hari` bervariasi antarpaket pada tanggal yang sama; volume kemarin memakai median per hari.
- Model dan threshold dipilih pada validation yang sama, sehingga angka validation cenderung sedikit optimis.
- Periode test pernah dievaluasi tim dengan model lain sebelumnya (lihat notebook bagian 10).

## Pembagian kerja dan penggunaan AI

_Diisi oleh masing-masing anggota sesuai kontribusi nyata._
