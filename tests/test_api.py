import importlib.util
import unittest

from pydantic import ValidationError

from app.core import Paket, Prediktor

VALID = dict(jam_pickup=14, kota_tujuan='Surabaya', jarak_km=678, jumlah_transit=2, layanan='REG', janji_hari=3,
             berat_kg=2, metode_bayar='COD', cuaca_saat_pickup='hujan', volume_hub_kemarin=5300)
TIDAK_WAJAR = [('berat_kg', -1), ('berat_kg', 1340), ('jam_pickup', 25), ('layanan', 'SAME DAY'),
               ('kota_tujuan', 'Tokyo'), ('jumlah_transit', 0), ('janji_hari', 0), ('volume_hub_kemarin', -5),
               ('jarak_km', 0), ('jam_pickup', 'abc'), ('terlambat', 1), ('jarak_km', float('nan'))]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = Prediktor()

    def test_prediksi_valid(self):
        hasil = self.pred.prediksi(Paket(**VALID))
        self.assertTrue(0 <= hasil['probabilitas_terlambat'] <= 1)
        self.assertIn(hasil['rekomendasi'], ['AMAN', 'KABARI PELANGGAN', 'PRIORITASKAN'])
        self.assertEqual(hasil['diisi_otomatis'], [])

    def test_input_tidak_wajar_ditolak(self):
        for kolom, nilai in TIDAK_WAJAR:
            with self.assertRaises(ValidationError, msg=kolom):
                Paket(**{**VALID, kolom: nilai})
        with self.assertRaises(ValidationError):
            Paket()

    def test_kolom_opsional_boleh_kosong(self):
        kosong = {**VALID, 'jarak_km': None, 'berat_kg': None, 'cuaca_saat_pickup': None, 'volume_hub_kemarin': None}
        hasil = self.pred.prediksi(Paket(**kosong))
        self.assertEqual(set(hasil['diisi_otomatis']), {'jarak_km', 'berat_kg', 'cuaca_saat_pickup', 'volume_hub_kemarin'})
        self.assertTrue(0 <= hasil['probabilitas_terlambat'] <= 1)

    def test_tiga_tingkat_rekomendasi(self):
        rendah = self.pred.prediksi(Paket(**{**VALID, 'kota_tujuan': 'Cirebon', 'layanan': 'HEMAT', 'jam_pickup': 9,
                                             'janji_hari': 5, 'jumlah_transit': 1, 'jarak_km': 130,
                                             'metode_bayar': 'non_COD', 'cuaca_saat_pickup': 'cerah',
                                             'volume_hub_kemarin': 5000}))
        tinggi = self.pred.prediksi(Paket(**{**VALID, 'kota_tujuan': 'Makassar', 'layanan': 'EXPRESS', 'jam_pickup': 20,
                                             'janji_hari': 2, 'jumlah_transit': 4, 'jarak_km': 1500,
                                             'cuaca_saat_pickup': 'hujan_lebat', 'volume_hub_kemarin': 11000}))
        self.assertEqual(rendah['rekomendasi'], 'AMAN')
        self.assertEqual(tinggi['rekomendasi'], 'PRIORITASKAN')

    def test_sama_dengan_pipeline_training(self):
        """Input aplikasi diolah sama persis dengan baris data validation."""
        import pandas as pd
        import train
        from src.features import KOLOM_FINAL, buat_fitur, hitung_volume_kemarin, isi_kosong, rapikan
        df, kerja, _ = train.pisah_waktu(pd.read_csv(train.ROOT / 'data' / 'raw.csv'))
        k = rapikan(kerja)
        baris = buat_fitur(isi_kosong(k[k.tanggal_pickup >= '2026-02-01'].head(50), self.pred.patokan),
                           hitung_volume_kemarin(df))
        for _, r in baris.iterrows():
            data = {c: (None if pd.isna(r[c]) else r[c]) for c in Paket.model_fields if c in r}
            data['volume_hub_kemarin'] = float(r['volume_hub_kemarin'])
            if data['cuaca_saat_pickup'] == 'tidak_diketahui':
                data['cuaca_saat_pickup'] = None
            p_app = self.pred.prediksi(Paket(**data))['probabilitas_terlambat']
            p_train = self.pred.model.predict_proba(baris.loc[[_], KOLOM_FINAL])[0, 1]
            self.assertLessEqual(abs(p_app - p_train), 5.1e-5)


@unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'), 'fastapi/httpx belum terpasang')
class ApiTests(unittest.TestCase):
    def test_endpoint(self):
        from fastapi.testclient import TestClient
        from app.api.main import app
        with TestClient(app) as c:
            self.assertEqual(c.get('/health').status_code, 200)
            self.assertIn('Prediksi', c.get('/').text)
            self.assertEqual(c.post('/predict', json=VALID).status_code, 200)
            for kolom, nilai in TIDAK_WAJAR[:-1]:
                self.assertEqual(c.post('/predict', json={**VALID, kolom: nilai}).status_code, 422, kolom)
            self.assertEqual(c.post('/predict', json={}).status_code, 422)
            self.assertEqual(c.post('/predict', content='bukan json').status_code, 422)


if __name__ == '__main__':
    unittest.main()
