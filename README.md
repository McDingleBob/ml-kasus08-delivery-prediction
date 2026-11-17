# Prediksi Keterlambatan Paket — Kasus 08

**Demo pendidikan nonoperasional, menggunakan data sintetis.** Anggota: Bob Nolen Barnabas Munthe (2406157), Muhammad Riyu Saputra (NIM belum terverifikasi), kelas 5B. Pembagian kerja dan commit anggota harus diisi berdasarkan kontribusi nyata; draf ini dibantu AI. URL repositori, aplikasi, dan video belum tersedia.

Sumber data dan brief: https://github.com/nurhikam/mkb-ml-project/tree/main/kasus/08-keterlambatan-paket . Sheet kelas menyatakan nomor topik 8; pencocokan dengan gambar daftar topik belum terverifikasi.

## Masalah dan data
CS dan operasional hub Bandung fiktif menerima probabilitas keterlambatan dan rekomendasi. 40.500 baris data sintetis, 500 duplikat persis dihapus, 40.000 unik. Jarak kosong pada raw 813, cuaca kosong 406. Pada data unik 1.274 berat nonpositif atau di luar lingkup demo 100 kg; pada training dijadikan missing dan diimputasi median train. Batas 100 kg bukan ketentuan brief atau bukti satuan salah; API menolak nilai <=0 atau >100 dengan HTTP 422, tetapi menerima null dan memakai imputasi train.

2025 train 26.633, Jan–Mar 2026 validation 6.623, Apr–Jun 2026 test 6.744. Imputasi dan transformasi di-fit pada train. Total volume hub harian tidak digunakan karena belum tersedia saat pickup. Transit dianggap terencana saat pickup, tetapi brief tidak memastikan ini: jika ternyata realisasi, model tidak layak untuk penggunaan pickup dan perlu eksperimen prospektif dan holdout baru. Data sintetis bukan bukti manfaat operasional.

## Hasil frozen
PR-AUC menggunakan average precision sklearn. Baseline regresi logistik AP validation 0.409362, test 0.360007; HGB terpilih AP validation 0.422425, test 0.367870. Threshold 0.06603616610212668 dipilih pada validation dengan biaya FN Rp50.000 dan FP Rp5.000, tanpa refit. Test frozen: precision 0.1441, recall 0.9904, alarm 95.70% paket; biaya simulasi 3.29% lebih rendah dari selalu memberi alarm. Ini manfaat lemah, bukan klaim penghematan nyata. Inventaris awal pernah melihat statistik global raw termasuk prevalence, jadi holdout tidak dapat diklaim tersegel sempurna.

## Notebook dan artefak
`notebooks/01_pipeline.ipynb` adalah versi normal yang dapat dibaca, melatih dan menilai ulang TRAIN/VALID saja. Hasilnya merupakan reproduksi terpisah, **bukan file model frozen yang digunakan API**, dan tidak mengevaluasi ulang TEST. Nilai TEST adalah hasil historis dari `artifacts/metadata.json` dan `artifacts/TEST_EVALUATED_ONCE.json`. Model API `artifacts/model.joblib` beserta metadata dan receipt disalin byte-identik dari artefak frozen. Dataset publik resmi di `data/raw.csv`; notebook mengunduh versi resmi dan memverifikasi SHA-256. `src/features.py` adalah transformasi untuk model frozen saat training dan prediksi; notebook memperlihatkan alur ekuivalen secara eksplisit, bukan membuktikan objek hasil rerun identik byte-per-byte dengan model frozen.

## Menjalankan API
Python 3.11 dan versi requirements yang disertakan.
```sh
pip install -r requirements.txt
uvicorn app.api.main:app --host 127.0.0.1 --port 7860
```
Buka `http://127.0.0.1:7860/docs`, pilih POST /predict, Try it out, isi JSON dan Execute. GET /health menunjukkan status. Swagger merupakan antarmuka web API; belum diuji sebagai deployment publik di salinan ini.

Contoh input valid:
```json
{"jam_pickup": 10, "kota_tujuan": "Jakarta", "jarak_km": 150, "jumlah_transit": 1, "layanan": "REG", "janji_hari": 2, "berat_kg": 2, "metode_bayar": "COD", "cuaca_saat_pickup": "cerah"}
```
Ubah `berat_kg` menjadi -1 untuk melihat HTTP 422; null diizinkan. `AMAN` bukan jaminan tepat waktu. Jangan mengevaluasi ulang test atau menghapus guard `train.py --finalize`.

## Status sebelum pengumpulan
Repo publik, akses collaborator dan commit nyata kedua anggota, aplikasi daring yang diuji termasuk ponsel, laporan yang diselaraskan dengan URL riil, serta video kedua anggota masih diperlukan. AI membantu pemulihan sumber, penulisan kode, eksperimen dan draf; anggota harus memahami setiap baris dan mencatat kontribusi mereka yang sebenarnya. Belum ada pengumpulan.
