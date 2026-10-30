import pandas as pd,numpy as np,json
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
P=pd.read_pickle('preds.pkl'); may,te=P['may'],P['test']
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def fit(kind,p,y):
    if kind=='iso': m=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p,y); return m, lambda q:m.predict(q)
    m=LogisticRegression(C=1e6).fit(lg(p)[:,None],y); return m, lambda q:m.predict_proba(lg(q)[:,None])[:,1]
def ece(p,y,b=10):
    q=pd.qcut(p,b,labels=False,duplicates='drop'); df=pd.DataFrame(dict(p=p,y=y,q=q)).groupby('q').agg(p=('p','mean'),y=('y','mean'),n=('y','size'))
    return float((df.n*(df.p-df.y).abs()).sum()/df.n.sum())
def mets(p,y): return dict(brier=brier_score_loss(y,p),logloss=log_loss(y,p),auc=roc_auc_score(y,p),ece=ece(p,y),mean_pred=float(p.mean()),obs=float(y.mean()))
R={}
# selection: May 1-15 fit, May 16-31 held out
t=pd.to_datetime(may.t,utc=True); a=may[t<pd.Timestamp('2026-05-16',tz='UTC')]; b=may[t>=pd.Timestamp('2026-05-16',tz='UTC')]
R['select_n']=[len(a),len(b)]
R['select_uncal']=mets(b.p.values,b.y.values)
for k in ['iso','platt']:
    _,f=fit(k,a.p.values,a.y.values); R['select_'+k]=mets(f(b.p.values),b.y.values)
# final on full May -> test
R['test_uncal']=mets(te.p.values,te.y.values); cal={}
for k in ['iso','platt']:
    m,f=fit(k,may.p.values,may.y.values); cal[k]=(m,f); R['test_'+k]=mets(f(te.p.values),te.y.values)
print(json.dumps(R,indent=1))
json.dump(R,open('cal_metrics.json','w'),indent=1)
