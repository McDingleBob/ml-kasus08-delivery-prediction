import unittest

import numpy as np
import pandas as pd

from src.features import KOLOM_FINAL, buat_fitur, hitung_volume_kemarin, isi_kosong, rapikan

PATOKAN = {'jarak_per_kota': {'Surabaya': 678.0}, 'median_berat': 1.34}


class FeatureTests(unittest.TestCase):
    def test_rapikan_ejaan_kode_error_dan_gram(self):
        raw = pd.DataFrame({'kota_tujuan': [' Sby', 'DKI Jakarta', 'bali', 'malang'],
                            'layanan': ['exp', 'Express', 'reg ', 'hemat'],
                            'berat_kg': [-1, 1340.0, 2.5, 50.0]})
        out = rapikan(raw)
        self.assertEqual(out.kota_tujuan.tolist(), ['Surabaya', 'Jakarta', 'Denpasar', 'Malang'])
        self.assertEqual(out.layanan.tolist(), ['EXPRESS', 'EXPRESS', 'REG', 'HEMAT'])
        self.assertTrue(np.isnan(out.berat_kg[0]))
        self.assertAlmostEqual(out.berat_kg[1], 1.34)      # >50 dianggap gram
        self.assertEqual(out.berat_kg[3], 50.0)             # 50 tetap kg
        self.assertEqual(raw.berat_kg[0], -1)               # input asli tidak diubah

    def test_isi_kosong_memakai_patokan_train(self):
        d = pd.DataFrame({'kota_tujuan': ['Surabaya', 'Medan'], 'jarak_km': [np.nan, np.nan],
                          'berat_kg': [np.nan, 3.0], 'cuaca_saat_pickup': [None, 'hujan']})
        out = isi_kosong(d, PATOKAN)
        self.assertEqual(out.jarak_km[0], 678.0)
        self.assertTrue(np.isnan(out.jarak_km[1]))          # kota tanpa patokan -> imputer pipeline
        self.assertEqual(out.berat_kg.tolist(), [1.34, 3.0])
        self.assertEqual(out.cuaca_saat_pickup.tolist(), ['tidak_diketahui', 'hujan'])

    def test_fitur_buatan(self):
        d = pd.DataFrame({'janji_hari': [3, 1], 'jumlah_transit': [1, 3], 'jam_pickup': [17, 18],
                          'volume_hub_kemarin': [9000, 9001]})
        out = buat_fitur(d)
        self.assertEqual(out.kelonggaran_janji.tolist(), [2, -2])
        self.assertEqual(out.pickup_malam.tolist(), [0, 1])
        self.assertEqual(out.volume_kemarin_tinggi.tolist(), [0, 1])

    def test_volume_kemarin_hanya_dari_hari_sebelumnya(self):
        df = pd.DataFrame({'tanggal_pickup': ['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-03'],
                           'volume_hub_hari': [100, 300, 500, 9999]})
        vk = hitung_volume_kemarin(df)
        self.assertTrue(np.isnan(vk[pd.Timestamp('2025-01-01')]))
        self.assertEqual(vk[pd.Timestamp('2025-01-02')], 200)
        self.assertEqual(vk[pd.Timestamp('2025-01-03')], 500)   # bukan 9999 (hari itu sendiri)

    def test_target_id_dan_volume_hari_ini_bukan_fitur(self):
        for kolom in ['terlambat', 'id_resi', 'volume_hub_hari', 'tanggal_pickup']:
            self.assertNotIn(kolom, KOLOM_FINAL)


if __name__ == '__main__':
    unittest.main()
