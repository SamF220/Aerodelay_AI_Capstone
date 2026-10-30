import pandas as pd,numpy as np,json
from sklearn.linear_model import LogisticRegression
P=pd.read_pickle('preds.pkl'); may,te=P['may'],P['test']
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
m=LogisticRegression(C=1e6).fit(lg(may.p.values)[:,None],may.y.values); a,b=float(m.coef_[0,0]),float(m.intercept_[0])
pc=1/(1+np.exp(-(a*lg(te.p.values)+b)))
json.dump(dict(method='platt',a=a,b=b,fit_on='May 2026 validation month (105,160 flights), shipped classifier, no retraining',formula='p_cal = 1/(1+exp(-(a*logit(p)+b)))'),open('calibration.json','w'),indent=1)
print('a,b',a,b)
y=te.y.values
# deciles on uncalibrated ranking (same groups both)
q=pd.qcut(te.p.values,10,labels=False)
D=pd.DataFrame(dict(q=q,before=te.p.values,after=pc,actual=y)).groupby('q').mean().round(4)
D.index=D.index+1; print(D.to_string())
def bands(p):
    bb=np.select([p<0.2,p<0.35],['LOW','ELEVATED'],'HIGH'); df=pd.DataFrame(dict(b=bb,y=y,p=p)).groupby('b').agg(share=('y','size'),pred=('p','mean'),actual=('y','mean'))
    df['share']=df.share/len(y); return df.reindex(['LOW','ELEVATED','HIGH']).round(4)
print('BEFORE\n',bands(te.p.values).to_string()); print('AFTER\n',bands(pc).to_string())
print('mean',te.p.mean(),pc.mean(),y.mean())
json.dump(dict(deciles=D.reset_index().to_dict('list'),bands_before=bands(te.p.values).reset_index().to_dict('list'),bands_after=bands(pc).reset_index().to_dict('list')),open('cal_tables.json','w'),indent=1)
