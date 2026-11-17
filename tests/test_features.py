import unittest
import pandas as pd

class FeatureTests(unittest.TestCase):
    def test_normalizes_city_and_units_without_target_or_id(self):
        from src.features import prepare
        raw=pd.DataFrame([dict(kota_tujuan='Sby',layanan='EXP',berat_kg=-1,jarak_km=None,id_resi='secret',terlambat=1,tanggal_pickup='2025-01-01',jam_pickup=14,jumlah_transit=2,janji_hari=2,metode_bayar='COD',cuaca_saat_pickup='hujan',volume_hub_hari=5000)])
        out=prepare(raw)
        self.assertEqual(out.kota_tujuan.iloc[0],'Surabaya')
        self.assertEqual(out.layanan.iloc[0],'EXPRESS')
        self.assertTrue(pd.isna(out.berat_kg.iloc[0]))
        self.assertNotIn('terlambat',out)
        self.assertNotIn('id_resi',out)
        self.assertNotIn('volume_hub_hari',out)

if __name__=='__main__': unittest.main()
