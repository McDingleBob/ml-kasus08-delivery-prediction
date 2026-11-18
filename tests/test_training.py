import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import train

ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_pisah_waktu_dan_duplikat(self):
        d = pd.DataFrame({'id_resi': ['a', 'a', 'b', 'c'], 'tanggal_pickup': ['2025-12-31'] * 2 + ['2026-03-31', '2026-04-01']})
        df, kerja, test = train.pisah_waktu(d)
        self.assertEqual((len(df), len(kerja), len(test)), (3, 2, 1))
        self.assertTrue((test.tanggal_pickup >= '2026-04-01').all())
        self.assertTrue((kerja.tanggal_pickup < '2026-04-01').all())

    def test_threshold_biaya(self):
        y = np.array([0, 0, 1, 1]); p = np.array([.01, .2, .1, .9])
        t = train.pilih_threshold(y, p)
        self.assertLessEqual(train.hitung_biaya(y, p, t)['biaya (Rp)'], 5000)
        self.assertLess(train.hitung_biaya(y, p, t)['biaya (Rp)'], train.hitung_biaya(y, p, .5)['biaya (Rp)'])

    def test_latih_tidak_memakai_test_dan_hasil_sama_dengan_notebook(self):
        dipakai = []
        asli = train.isi_kosong
        def catat(data, patokan):
            dipakai.append(pd.to_datetime(data['tanggal_pickup']).max())
            return asli(data, patokan)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train, 'isi_kosong', catat):
            meta = train.latih(keluaran=tmp)
            self.assertTrue((Path(tmp) / 'model.joblib').exists())
        self.assertTrue(all(t < pd.Timestamp('2026-04-01') for t in dipakai))
        self.assertNotIn('test', json.dumps(meta['metrik_validation']).lower())
        self.assertEqual(meta['baris']['train'], 26633)
        self.assertEqual(meta['baris']['validation'], 6623)
        # angka notebook ML_Kasus08_Final.ipynb bagian 9
        self.assertEqual(meta['metrik_validation']['pr_auc_model'], 0.605)
        self.assertEqual(meta['metrik_validation']['pr_auc_baseline'], 0.409)
        self.assertEqual(meta['threshold_kabari'], 0.1)
        self.assertEqual(meta['metrik_validation']['biaya_model_rp'], 20_730_000)

    def test_metadata_repo_konsisten(self):
        meta = json.loads((ROOT / 'models' / 'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['nama_model'], 'Regresi logistik (C=0.1)')
        self.assertEqual(meta['threshold_kabari'], 0.1)


if __name__ == '__main__':
    unittest.main()
