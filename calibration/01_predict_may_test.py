import sys; sys.path.insert(0,'/workspace/colab/retrain')
from common import *; import joblib,pandas as pd,numpy as np
A=joblib.load('/workspace/gh_repo/aerodelay_artifacts.pkl'); E=joblib.load('/workspace/gh_repo/aerodelay_encoding_maps.pkl'); clf=joblib.load('/workspace/gh_repo/aerodelay_log_model.pkl')
F=A['model_feature_columns']; med=A['train_medians']
d=pd.read_parquet('/workspace/colab/retrain/data/train_table.parquet'); tr,va,te=split(d); print(len(tr),len(va),len(te))
out={}
for nm,x in [('may',va),('test',te)]:
    x=apply_enc(apply_safe(x,A),E)
    for c in F: x[c]=x[c].fillna(med[c])
    p=clf.predict_proba(x[F])[:,1]
    out[nm]=pd.DataFrame({'p':p,'y':x.delay_binary.values,'t':x.scheduled_departure_utc.values})
    print(nm,len(x),p.mean(),x.delay_binary.mean())
pd.to_pickle(out,'preds.pkl')
from sklearn.metrics import roc_auc_score; print('test auc',roc_auc_score(out['test'].y,out['test'].p))
