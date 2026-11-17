import unittest
from fastapi.testclient import TestClient

VALID=dict(jam_pickup=14,kota_tujuan='Surabaya',jarak_km=678,jumlah_transit=2,layanan='REG',janji_hari=3,berat_kg=2,metode_bayar='COD',cuaca_saat_pickup='hujan')
class ApiTests(unittest.TestCase):
    def test_prediction_health_and_invalid_contract(self):
        from app.api.main import app
        with TestClient(app) as c:
            self.assertEqual(c.get('/health').status_code,200)
            self.assertEqual(c.get('/').status_code,200)
            good=c.post('/predict',json=VALID)
            self.assertEqual(good.status_code,200)
            result=good.json()
            self.assertTrue(0<=result['probabilitas_terlambat']<=1)
            self.assertIn(result['rekomendasi'],['AMAN','KABARI PELANGGAN DAN PRIORITASKAN'])
            for key,value in [('berat_kg',-1),('jam_pickup',25),('layanan','SAME DAY'),('jumlah_transit',-1),('terlambat',1)]:
                self.assertEqual(c.post('/predict',json={**VALID,key:value}).status_code,422,key)
            self.assertEqual(c.post('/predict',json={**VALID,'jarak_km':None,'cuaca_saat_pickup':None}).status_code,200)
            self.assertEqual(c.post('/predict',json={}).status_code,422)

if __name__=='__main__': unittest.main()
