"""Targeted major-revision experiments; fixed settings, untouched test selection.

Run analyze.py first. Outputs live in results/revision/. Daily forecasts are issued
at midnight for 05--23h, with monthly frozen fits. RECENT alone updates from
recorded targets strictly before that midnight; its extra information is explicit.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from analyze import load, features, flight_series, estimate_pax, estimator, metrics
from revision_io import compress_outputs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'revision'
FOLDS = ['2024-09', '2025-03', '2025-04', '2025-05']
NAMES = {'Media storica ora × settimana':'MEAN', 'Calendario':'CAL',
         'Calendario + voli programmati':'SCHED', 'Calendario + pax stimati':'PAX',
         'Ridge calendario + pax stimati':'RIDGE',
         'Retrospettivo pax e orari effettivi':'ACTUAL-EXTRA'}


def expectation_level(f, cut):
    h = f[(f.BLOCK_TIME < cut) & f.BOUND.isin(['A','D']) & (f.pax > 0)]
    n = h.groupby(['FLIGHT_NUMBER','BOUND']).size()
    reliable = n[n >= 5].index
    known_routes = h.groupby(['ORIDES_CODE','BOUND']).size().index
    flight = pd.MultiIndex.from_frame(f[['FLIGHT_NUMBER','BOUND']]).isin(reliable)
    route = pd.MultiIndex.from_frame(f[['ORIDES_CODE','BOUND']]).isin(known_routes)
    return np.where(flight, 'flight', np.where(route, 'airport', 'global'))


def paired(pred):
    rows = []; rng = np.random.default_rng(42)
    for (typ, fold), z in pred.groupby(['type','fold']):
        p = z.pivot(index='time', columns='model', values='prediction')
        y = z.drop_duplicates('time').set_index('time').y.reindex(p.index)
        for ref in ['CAL','MEAN','SCHED','CAL-REG','RECENT','PAX']:
            for model in (['REAL-LOAD','REAL-TIME','REAL-BOTH'] if ref=='PAX' else ['SCHED','PAX']):
                if model == ref: continue
                daily = pd.DataFrame({'base':abs(y-p[ref]), 'model':abs(y-p[model])}).groupby(p.index.normalize()).sum()
                draws = rng.integers(0,len(daily),size=(1000,len(daily)))
                b = daily.base.to_numpy(); m = daily.model.to_numpy()
                gain = 100*(1-m[draws].sum(1)/b[draws].sum(1))
                rows.append(dict(type=typ,fold=fold,model=model,reference=ref,
                                 gain_pct=100*(1-m.sum()/b.sum()),low=np.quantile(gain,.025),high=np.quantile(gain,.975),days=len(daily),hours=len(y)))
    return pd.DataFrame(rows)


def directional(pred):
    rows=[]; rng=np.random.default_rng(43)
    for fold in FOLDS:
        z=pred[pred.fold==fold]
        panels={}
        for typ in ['DA APT','DA FS']:
            d=z[z.type==typ]; p=d.pivot(index='time',columns='model',values='prediction')
            panels[typ]=(p,d.drop_duplicates('time').set_index('time').y.reindex(p.index))
        common=panels['DA APT'][0].index.intersection(panels['DA FS'][0].index)
        for ref in ['CAL','MEAN','SCHED','CAL-REG','RECENT']:
            g=[]
            for typ in ['DA APT','DA FS']:
                p,y=panels[typ]; y=y.reindex(common);p=p.reindex(common)
                g.append(pd.DataFrame({'b':abs(y-p[ref]),'m':abs(y-p.PAX)}).groupby(common.normalize()).sum())
            draws=rng.integers(0,len(g[0]),size=(1000,len(g[0])))
            gains=[100*(1-d.m.sum()/d.b.sum()) for d in g]
            samples=[100*(1-d.m.to_numpy()[draws].sum(1)/d.b.to_numpy()[draws].sum(1)) for d in g]
            diff=samples[0]-samples[1]
            rows.append(dict(fold=fold,reference=ref,apt_gain=gains[0],fs_gain=gains[1],difference_pp=gains[0]-gains[1],low=np.quantile(diff,.025),high=np.quantile(diff,.975),common_hours=len(common),common_days=len(g[0])))
    return pd.DataFrame(rows)


def diagnostics(pred, calibration, quantiles):
    rows=[]
    for (typ,fold,model),z in pred.groupby(['type','fold','model']):
        peak=z.y>=z.peak_threshold
        e=z.prediction-z.y; pe=e[peak]
        rows.append(dict(type=typ,fold=fold,model=model,peak_n=int(peak.sum()),n=len(z),peak_bias=pe.mean(),peak_MAE=abs(pe).mean(),under_pct=100*(pe<0).mean(),error_p10=pe.quantile(.1),error_p50=pe.quantile(.5),error_p90=pe.quantile(.9)))
    pd.DataFrame(rows).to_csv(OUT/'peak_diagnostics.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibration.csv',index=False)
    pd.DataFrame(quantiles).to_csv(OUT/'quantile_metrics.csv',index=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    f,s,_=load()
    old=pd.read_csv(ROOT/'results/predictions.csv',parse_dates=['time'])
    old['model']=old.model.map(NAMES)
    preds=[old]; rows=[]; fallback=[]; recent_audit=[]; calibration=[]; quantiles=[]; cv=[]; qpreds=[]
    valid=s[(s.type!='SS')&s.hour.between(5,23)&s.time.between(f.BLOCK_TIME.min()+pd.Timedelta(days=1),f.BLOCK_TIME.max()-pd.Timedelta(days=1))].copy()
    for fold in FOLDS:
        cut=pd.Timestamp(fold);end=cut+pd.offsets.MonthBegin(1)
        f['estimated_pax']=estimate_pax(f,cut);level=expectation_level(f,cut)
        for bound in ['A','D']:
            mask=f.BOUND.eq(bound)&f.SCHEDULED_TIME.ge(cut-pd.Timedelta(hours=3))&f.SCHEDULED_TIME.lt(end+pd.Timedelta(hours=4))
            n=int(mask.sum())
            for kind in ['flight','airport','global']:
                fallback.append(dict(fold=fold,bound=bound,level=kind,n=int((level[mask]==kind).sum()),total=n,pct=100*(level[mask]==kind).mean()))
        series={name:flight_series(f,clock,pax) for name,clock,pax in [
            ('PAX','SCHEDULED_TIME','estimated_pax'),('REAL-LOAD','SCHEDULED_TIME','pax'),
            ('REAL-TIME','BLOCK_TIME','estimated_pax'),('REAL-BOTH','BLOCK_TIME','pax')]}
        for typ in ['DA APT','DA FS']:
            d=valid[valid.type==typ].reset_index(drop=True);tr=d.time<cut;te=(d.time>=cut)&(d.time<end)
            y=d.loc[te,'y'];threshold=d.loc[tr,'y'].quantile(.9)
            def record(name,p):
                p=np.maximum(0,p);out=d.loc[te,['time','type','y']].copy();out['prediction']=p;out['model']=name;out['fold']=fold;out['peak_threshold']=threshold;preds.append(out)
                rows.append(dict(type=typ,fold=fold,model=name,train_n=int(tr.sum()),peak_threshold=threshold,**metrics(y,p,threshold)))
                return p
            for name in ['REAL-LOAD','REAL-TIME','REAL-BOTH']:
                x=features(d.time,series[name]);reg=estimator(name);reg.fit(x.loc[tr],d.loc[tr,'y']);record(name,reg.predict(x.loc[te]))
            x=features(d.time,{});x['hour_weekday']=x.hour.astype(str)+'_'+x.dow.astype(str)
            x['annual_sin2']=np.sin(4*np.pi*d.time.dt.dayofyear/365.25);x['annual_cos2']=np.cos(4*np.pi*d.time.dt.dayofyear/365.25)
            def calendar_reg(alpha):
                return make_pipeline(ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['hour','dow','hour_weekday','month'])],remainder=StandardScaler()),Ridge(alpha=alpha))
            xt=x.loc[tr].reset_index(drop=True);yt=d.loc[tr,'y'].reset_index(drop=True);scores={}
            for alpha in [.1,1,10,100,1000]:
                errors=[]
                for inner_train,inner_test in TimeSeriesSplit(n_splits=3).split(xt):
                    cr=calendar_reg(alpha);cr.fit(xt.iloc[inner_train],yt.iloc[inner_train]);errors.append(mean_absolute_error(yt.iloc[inner_test],np.maximum(0,cr.predict(xt.iloc[inner_test]))))
                scores[alpha]=float(np.mean(errors));cv.append(dict(type=typ,fold=fold,alpha=alpha,inner_MAE=scores[alpha],latest_validation_time=str(d.loc[tr,'time'].max())))
            chosen=min(scores,key=scores.get);reg=calendar_reg(chosen)
            reg.fit(x.loc[tr],d.loc[tr,'y']);record('CAL-REG',reg.predict(x.loc[te]))
            frozen=d.loc[tr].groupby(['hour','dow']).y.mean();global_mean=d.loc[tr,'y'].mean();pr=[]
            for row in d.loc[te].itertuples():
                issue=row.time.normalize();pool=d[(d.time<issue)&(d.time>=issue-pd.Timedelta(days=28))&(d.hour==row.hour)&(d.dow==row.dow)]
                n=len(pool);pr.append(pool.y.mean() if n else frozen.get((row.hour,row.dow),global_mean))
                recent_audit.append(dict(type=typ,fold=fold,time=row.time,issue_time=issue,recent_n=n,fallback=(n==0),latest_source_time=pool.time.max() if n else pd.NaT))
            record('RECENT',np.array(pr))
            x=features(d.time,series['PAX']);reg=estimator('PAX');reg.fit(x.loc[tr],d.loc[tr,'y'])
            point=np.maximum(0,reg.predict(x.loc[te]));reference=old[(old.fold==fold)&(old.type==typ)&(old.model=='PAX')].sort_values('time').prediction.to_numpy()
            assert np.allclose(point,reference,atol=1e-10)
            train_p=reg.predict(x.loc[tr]);edges=np.unique(np.quantile(train_p,np.linspace(0,1,6)));edges[0]=-np.inf;edges[-1]=np.inf
            bins=pd.cut(point,edges,include_lowest=True)
            for label,g in pd.DataFrame({'y':y.to_numpy(),'p':point,'bin':bins}).groupby('bin',observed=True):
                calibration.append(dict(type=typ,fold=fold,bin=str(label),n=len(g),predicted_mean=g.p.mean(),observed_mean=g.y.mean(),bias=(g.p-g.y).mean()))
            qr=HistGradientBoostingRegressor(loss='quantile',quantile=.9,max_iter=220,max_leaf_nodes=15,max_depth=4,min_samples_leaf=35,l2_regularization=10,learning_rate=.06,early_stopping=False,random_state=42)
            qr.fit(x.loc[tr],d.loc[tr,'y']);q=np.maximum(0,qr.predict(x.loc[te]));peak=y.to_numpy()>=threshold
            quantiles.append(dict(type=typ,fold=fold,n=len(y),q90_coverage=100*np.mean(y<=q),peak_n=int(peak.sum()),peak_coverage=100*np.mean(y.to_numpy()[peak]<=q[peak]),q90_pinball=mean_pinball_loss(y,q,alpha=.9),point_pinball=mean_pinball_loss(y,point,alpha=.9)))
            qout=d.loc[te,['time','type','y']].copy();qout['fold']=fold;qout['q90']=q
            qpreds.append(qout)
            print(f'Revision {fold} {typ} complete',flush=True)
    pred=pd.concat(preds,ignore_index=True);pred.to_csv(OUT/'predictions.csv',index=False)
    pd.concat(qpreds,ignore_index=True).to_csv(OUT/'quantile_predictions.csv',index=False)
    oldmet=pd.read_csv(ROOT/'results/model_metrics.csv');oldmet['model']=oldmet.model.map(NAMES)
    pd.concat([oldmet,pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/'model_metrics.csv',index=False)
    paired(pred).to_csv(OUT/'paired_gains.csv',index=False);directional(pred).to_csv(OUT/'directional_gains.csv',index=False)
    pd.DataFrame(fallback).to_csv(OUT/'passenger_fallback.csv',index=False);pd.DataFrame(recent_audit).to_csv(OUT/'recent_audit.csv',index=False)
    pd.DataFrame(cv).to_csv(OUT/'calendar_cv.csv',index=False)
    diagnostics(pred,calibration,quantiles)
    meta={'issue':'00:00 local civil time on target day','horizons_hours':[5,23],'monthly_fit':'First day 00:00; all fitting inputs strictly earlier','passenger_estimates':'Frozen monthly','recent_window_days':28,'recent_update':'Daily; recorded targets strictly before issue; frozen MEAN fallback','schedule_status':'Retrospective movement selection; cancellation completeness unknown','bootstrap':{'replicates':1000,'unit':'day','seed':42,'directional_seed':43,'between_day_dependence':'not modeled'},'ablation':'same 24 features; no low-cost/domestic terms'}
    (OUT/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'quantile_predictions.csv').unlink(missing_ok=True)
    main()
    compress_outputs()
