"""Verify targeted experiments, online information cutoffs, and saved diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_pinball_loss
from revision_io import read_result
R=Path(__file__).resolve().parent/'results/revision'
p=read_result('predictions.csv',parse_dates=['time']);m=pd.read_csv(R/'model_metrics.csv')
assert not p.duplicated(['time','type','model','fold']).any()
for (typ,fold,model),z in p.groupby(['type','fold','model']):
    a=m[(m.type==typ)&(m.fold==fold)&(m.model==model)].iloc[0]
    assert len(z)==a.n and z.time.dt.strftime('%Y-%m').eq(fold).all()
    assert np.isclose(mean_absolute_error(z.y,z.prediction),a.MAE)
    assert z.prediction.ge(0).all()
    peer=p[(p.type==typ)&(p.fold==fold)&(p.model=='PAX')]
    assert set(z.time)==set(peer.time)
a=read_result('recent_audit.csv',parse_dates=['time','issue_time','latest_source_time'])
assert a.issue_time.eq(a.time.dt.normalize()).all()
assert (a.latest_source_time.dropna()<a.loc[a.latest_source_time.notna(),'issue_time']).all()
assert a.fallback.eq(a.recent_n.eq(0)).all()
c=pd.read_csv(R/'calendar_cv.csv',parse_dates=['latest_validation_time'])
assert (c.latest_validation_time<pd.to_datetime(c.fold)).all()
g=pd.read_csv(R/'paired_gains.csv')
assert (g.low<=g.high).all()
for row in g.itertuples():
    a=m[(m.type==row.type)&(m.fold==row.fold)].set_index('model')
    assert np.isclose(row.gain_pct,100*(1-a.loc[row.model,'MAE']/a.loc[row.reference,'MAE']))
direction=pd.read_csv(R/'directional_gains.csv')
for row in direction.itertuples():
    z=p[p.fold==row.fold];panels={}
    for typ in ['DA APT','DA FS']:
        a=z[z.type==typ];panels[typ]=a.pivot(index='time',columns='model',values='prediction')
    common=panels['DA APT'].index.intersection(panels['DA FS'].index);gains=[]
    for typ in ['DA APT','DA FS']:
        y=z[z.type==typ].drop_duplicates('time').set_index('time').y.reindex(common);a=panels[typ].reindex(common)
        gains.append(100*(1-abs(y-a.PAX).sum()/abs(y-a[row.reference]).sum()))
    assert len(common)==row.common_hours and common.normalize().nunique()==row.common_days
    assert np.isclose(gains[0]-gains[1],row.difference_pp)
peak=pd.read_csv(R/'peak_diagnostics.csv')
for row in peak.itertuples():
    z=p[(p.type==row.type)&(p.fold==row.fold)&(p.model==row.model)];ispeak=z.y>=z.peak_threshold;e=(z.prediction-z.y)[ispeak]
    assert len(e)==row.peak_n and np.isclose(e.mean(),row.peak_bias)
    assert np.isclose(100*(e<0).mean(),row.under_pct)
q=read_result('quantile_predictions.csv');qm=pd.read_csv(R/'quantile_metrics.csv')
for (typ,fold),z in q.groupby(['type','fold']):
    a=qm[(qm.type==typ)&(qm.fold==fold)].iloc[0]
    assert len(z)==a.n and np.isclose(100*(z.y<=z.q90).mean(),a.q90_coverage)
    assert np.isclose(mean_pinball_loss(z.y,z.q90,alpha=.9),a.q90_pinball)
summary={'prediction_rows':len(p),'models':int(p.model.nunique()),'folds':4,'paired_gain_rows':len(g),'checks':'metrics, identical evaluation hours, recent-data cutoffs, inner validation ordering, quantile coverage'}
(R/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
