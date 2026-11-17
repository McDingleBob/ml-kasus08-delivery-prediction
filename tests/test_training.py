import unittest
import numpy as np

class TrainingTests(unittest.TestCase):
    def test_build_uses_validation_and_does_not_evaluate_test_by_default(self):
        from train import build
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            meta=build(Path(tmp),finalize=False)
            self.assertEqual(meta['test_evaluations'],0)
            self.assertNotIn('test_metrics',meta)
            self.assertIn('logistic_regression',meta['validation'])
            self.assertTrue((Path(tmp)/'model.joblib').exists())

    def test_temporal_split_keeps_newest_rows_for_final_test(self):
        import pandas as pd
        from train import temporal_split
        d=pd.DataFrame({'tanggal_pickup':['2025-12-31','2026-01-01','2026-03-31','2026-04-01']})
        a,b,c=temporal_split(d)
        self.assertEqual((len(a),len(b),len(c)),(1,2,1))
        self.assertTrue(set(a.index).isdisjoint(c.index))

    def test_business_threshold_reduces_weighted_error(self):
        from train import choose_threshold, metrics
        y=np.array([0,0,1,1]); p=np.array([.01,.2,.1,.9])
        t=choose_threshold(y,p)
        self.assertEqual(metrics(y,p,t)['cost_rupiah'],5000)
        self.assertLess(metrics(y,p,t)['cost_rupiah'],metrics(y,p,.5)['cost_rupiah'])

if __name__=='__main__': unittest.main()
