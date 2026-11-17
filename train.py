from sklearn.metrics import average_precision_score,confusion_matrix,precision_score,recall_score
import numpy as np
import pandas as pd

def temporal_split(d):
    dates=pd.to_datetime(d['tanggal_pickup'],errors='raise')
    return (d.loc[dates<'2026-01-01'].copy(),d.loc[(dates>='2026-01-01')&(dates<'2026-04-01')].copy(),d.loc[dates>='2026-04-01'].copy())

def metrics(y,p,threshold):
    pred=np.asarray(p)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(pr_auc_average_precision=float(average_precision_score(y,p)),threshold=float(threshold),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),cost_rupiah=int(50000*fn+5000*fp))

def choose_threshold(y,p):
    # Evaluate every distinct validation score in O(n log n), including all-negative.
    y=np.asarray(y); p=np.asarray(p); order=np.argsort(-p,kind='stable'); p=p[order]; y=y[order]
    tp=np.cumsum(y); fp=np.cumsum(1-y); costs=50000*(y.sum()-tp)+5000*fp
    ends=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1]
    best=ends[np.argmin(costs[ends])]
    if 50000*y.sum()<costs[best]: return float(np.nextafter(p[0],np.inf))
    return float(p[best])


def build(output,finalize=False):
    from pathlib import Path
    from datetime import datetime
    import json,hashlib,platform,importlib.metadata,joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer,OneHotEncoder,StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src.features import prepare,FEATURES,NUMERIC,CATEGORICAL
    output=Path(output)
    if (output/'TEST_EVALUATED_ONCE.json').exists():
        raise RuntimeError('Final audit exists: use a separate preview output; never overwrite final metadata.')
    root=Path(__file__).resolve().parent
    source=root/'data'/'raw.csv'
    raw=pd.read_csv(source)
    duplicates=int(raw.duplicated().sum())
    d=raw.drop_duplicates().copy()
    a,v,t=temporal_split(d)
    assert a.id_resi.is_unique and v.id_resi.is_unique and t.id_resi.is_unique
    assert set(a.id_resi).isdisjoint(t.id_resi) and set(v.id_resi).isdisjoint(t.id_resi)
    models={}; validation={}
    for name,estimator in [('logistic_regression',LogisticRegression(max_iter=2000,random_state=42)),('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=5,learning_rate=.07,random_state=42))]:
        pre=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),NUMERIC),
            ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATEGORICAL)
        ])
        pipe=Pipeline([('clean',FunctionTransformer(prepare,validate=False)),('preprocess',pre),('model',estimator)])
        pipe.fit(a.drop(columns=['terlambat']),a.terlambat)
        prob=pipe.predict_proba(v.drop(columns=['terlambat']))[:,1]
        threshold=choose_threshold(v.terlambat.to_numpy(),prob)
        validation[name]=metrics(v.terlambat,prob,threshold)
        models[name]=pipe
    selected=max(validation,key=lambda n:validation[n]['pr_auc_average_precision'])
    threshold=validation[selected]['threshold']
    metadata=dict(created_at=datetime.now().astimezone().isoformat(),version='0.1.0-review',case='08-keterlambatan-paket',source_url='https://github.com/nurhikam/mkb-ml-project',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),features=FEATURES,excluded_features={'id_resi':'identifier, not a causal predictor','tanggal_pickup':'split only; no engineered features in baseline','volume_hub_hari':'full-day count not known at pickup; excluded to prevent future-information leakage'},selection_rule='highest validation average precision (non-interpolated PR-AUC), fixed two-model comparison',selected_model=selected,threshold=threshold,false_negative_cost=50000,false_positive_cost=5000,validation=validation,split={key:dict(n=len(part),start=str(part.tanggal_pickup.min()),end=str(part.tanggal_pickup.max())) for key,part in [('train',a),('validation',v),('test',t)]},cleaning={'raw_rows':len(raw),'duplicate_rows_removed':duplicates,'unique_rows':len(d),'distance_missing_raw':int(raw.jarak_km.isna().sum()),'weather_missing_raw':int(raw.cuaca_saat_pickup.isna().sum()),'invalid_weight_unique':int(((d.berat_kg<=0)|(d.berat_kg>100)).sum()),'weight_policy':'nonpositive or >100kg -> missing, then train median; >100kg limit is project scope assumption, not proof of grams'},versions={n:importlib.metadata.version(n) for n in ['pandas','numpy','scikit-learn','joblib','fastapi','pydantic','uvicorn']},python=platform.python_version(),test_evaluations=0,audit_note='Initial source inventory described full raw data including overall target prevalence before splitting. No test model scores used in model/threshold selection. Not a perfectly sealed holdout audit.')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    # Preview/test-suite runs NEVER calculate test scores.
    if finalize:
        receipt=output/'TEST_EVALUATED_ONCE.json'
        if receipt.exists(): raise RuntimeError('Test already evaluated. Do not reopen or retune against holdout.')
        metadata['test_metrics']={name:metrics(t.terlambat,model.predict_proba(t.drop(columns=['terlambat']))[:,1],validation[name]['threshold']) for name,model in models.items()}
        metadata['test_evaluations']=1
        receipt.write_text(json.dumps({'created_at':metadata['created_at'],'selected_before_test':selected,'threshold_before_test':threshold,'test_metrics':metadata['test_metrics'],'source_sha256':metadata['source_sha256']},indent=2),encoding='utf-8')
    joblib.dump(models[selected],output/'model.joblib')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (root/'data'/'clean.csv').parent.mkdir(exist_ok=True)
    pd.concat([prepare(d),d[['id_resi','tanggal_pickup','terlambat']]],axis=1).to_csv(root/'data'/'clean.csv',index=False)
    return metadata

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser(); parser.add_argument('--finalize',action='store_true'); args=parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parent/'artifacts',args.finalize),indent=2))

