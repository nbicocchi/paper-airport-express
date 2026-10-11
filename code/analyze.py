"""Read-only analysis of the pinned BLQ / Marconi Express data. No Excel writes.
All models use chronological splits and fixed hyperparameters. See report.md.
"""
from pathlib import Path
import json, os, sys, platform
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
import sklearn

from data_sources import DATA_ROOT, verify_sources

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize':(11,5),'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'font.size':10,'savefig.dpi':160})
LABEL={'DA APT':'Aeroporto → stazione','DA FS':'Stazione → aeroporto','SS':'Servizi sostitutivi'}
COLORS=['#167d9a','#e89237','#8f64b4']
def save(name):
    plt.tight_layout(); plt.savefig(FIG/name,bbox_inches='tight'); plt.close()
def md(d):
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v,(float,np.floating)): return f'{v:.2f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
def log(s): print(s,flush=True)

def load():
    verify_sources()
    flights=[]; audit=[]; shuttle=[]
    for p in sorted((DATA_ROOT/'Aeroporto').glob('*.xlsx')):
        d=pd.read_excel(p); d['source']=p.name
        for c in ['SCHEDULED_TIME','BLOCK_TIME']: d[c]=pd.to_datetime(d[c],errors='coerce')
        for c in ['PAX_ON_BOARD','PAX_ON_BOARD_NO_TRANSIT','BAG_NUMBER']: d[c]=pd.to_numeric(d[c],errors='coerce')
        audit.append({'file':p.name,'rows':len(d),'first':str(d.BLOCK_TIME.min()),'last':str(d.BLOCK_TIME.max()),'missing_bound':int((~d.BOUND.isin(['A','D'])).sum()),'duplicates_exact':int(d.drop(columns='source').duplicated().sum())})
        flights.append(d)
    f=pd.concat(flights,ignore_index=True)
    for p in sorted((DATA_ROOT/'Marconi Express').glob('*.xlsx')):
        raw=pd.read_excel(p); d=raw.iloc[:,:3].copy(); d.columns=['date','hour','y']
        d['date']=pd.to_datetime(d.date,errors='coerce'); d['hour']=pd.to_numeric(d.hour,errors='coerce'); d['y']=pd.to_numeric(d.y,errors='coerce')
        valid=d.date.notna() & d.hour.between(0,23) & (d.hour%1==0) & d.y.notna() & (d.y>=0)
        typ='SS' if p.name.startswith('SS') else ('DA APT' if p.name.startswith('DA APT') else 'DA FS')
        clean=d[valid].copy(); clean['type']=typ; clean['source']=p.name; clean['time']=clean.date+pd.to_timedelta(clean.hour,unit='h')
        clean['month']=clean.date.dt.strftime('%Y-%m')
        audit.append({'file':p.name,'rows':len(raw),'valid_rows':len(clean),'excluded_rows':int((~valid).sum()),'excluded_y':float(d.loc[~valid,'y'].sum()),'first':str(clean.date.min()),'last':str(clean.date.max()),'days':clean.date.nunique(),'zeros':int((clean.y==0).sum()),'duplicates_exact':int(clean.duplicated(['time']).sum())})
        shuttle.append(clean)
    s=pd.concat(shuttle,ignore_index=True).sort_values('time')
    pd.DataFrame(audit).to_csv(OUT/'source_audit.csv',index=False)
    assert not s.duplicated(['type','time']).any(), 'duplicate shuttle keys'
    # Exact duplicate flight rows are reported, not removed without a movement ID.
    f['pax']=f.PAX_ON_BOARD_NO_TRANSIT
    f['date']=f.BLOCK_TIME.dt.normalize(); f['month']=f.BLOCK_TIME.dt.strftime('%Y-%m')
    f['hour']=f.BLOCK_TIME.dt.hour; f['year']=f.BLOCK_TIME.dt.year
    s['dow']=s.date.dt.dayofweek
    s.to_csv(OUT/'shuttle_clean.csv',index=False)
    log(f'Loaded {len(f)} flights, {len(s)} valid shuttle rows')
    return f,s,pd.DataFrame(audit)

def describe(f,s):
    annual=f.groupby('year').agg(records=('pax','size'),pax_no_transit=('pax','sum'),days=('date','nunique'),first=('date','min'),last=('date','max'))
    annual['arrivals']=f[f.BOUND=='A'].groupby('year').size(); annual['departures']=f[f.BOUND=='D'].groupby('year').size()
    annual['unclassified']=f[~f.BOUND.isin(['A','D'])].groupby('year').size()
    annual.fillna({'unclassified':0},inplace=True); annual.reset_index().to_csv(OUT/'flights_annual.csv',index=False)
    monthly=f.groupby(['month','BOUND'],dropna=False).agg(movements=('pax','size'),pax=('pax','sum')).reset_index(); monthly.to_csv(OUT/'flights_monthly.csv',index=False)
    sm=s.groupby(['month','type']).agg(rows=('y','size'),days=('date','nunique'),total=('y','sum'),mean_observed=('y','mean'),p95=('y',lambda x:x.quantile(.95)),max=('y','max')).reset_index()
    sm['hours_5_23']=sm.apply(lambda r: int(((s.month==r.month)&(s.type==r.type)&s.hour.between(5,23)).sum()),axis=1)
    sm['expected_5_23']=pd.to_datetime(sm.month).dt.days_in_month*19
    sm['coverage_5_23_pct']=100*sm.hours_5_23/sm.expected_5_23
    sm.to_csv(OUT/'shuttle_monthly.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(13,4.7))
    for j,metric in enumerate(['movements','pax']):
        piv=monthly[monthly.BOUND.isin(['A','D'])].pivot(index='month',columns='BOUND',values=metric)
        for i,c in enumerate(piv): ax[j].plot(pd.to_datetime(piv.index),piv[c],label={'A':'Arrivi','D':'Partenze'}[c],color=COLORS[i])
        ax[j].axvspan(pd.Timestamp('2025-07-01'),pd.Timestamp('2025-07-20'),color='#e89237',alpha=.15)
        ax[j].text(.98,.97,'Luglio 2025: solo 1–10',transform=ax[j].transAxes,ha='right',va='top',fontsize=8)
        ax[j].set_title('Movimenti mensili' if j==0 else 'Passeggeri mensili, esclusi transiti'); ax[j].legend(); ax[j].tick_params(axis='x',rotation=35)
    save('01_flights_monthly.png')
    v=f[f.BOUND.isin(['A','D'])]
    hourly=v.groupby(['BOUND','hour']).pax.sum().unstack(0)/f.date.nunique()
    hourly.rename(columns={'A':'Arrivi','D':'Partenze'}).plot(color=COLORS[:2]); plt.xlabel('Ora BLOCK_TIME'); plt.ylabel('Passeggeri medi per giorno'); plt.title('Profilo orario aereo sui giorni presenti nei file'); save('02_flights_hourly.png')
    countries=v.groupby('COUNTRY_NAME').pax.sum().nlargest(12).sort_values()
    countries.plot.barh(color=COLORS[0]); plt.xlabel('Passeggeri esclusi transiti, tutti i periodi disponibili'); plt.title('Paesi principali'); save('03_countries.png')
    fig,ax=plt.subplots(1,2,figsize=(13,4.7))
    for i,t in enumerate(['DA APT','DA FS']):
        sub=s[(s.type==t)&s.hour.between(5,23)]; g=sub.groupby('hour').y
        ax[0].plot(g.mean().index,g.mean(),label=LABEL[t],color=COLORS[i]); ax[0].fill_between(g.mean().index,g.quantile(.1),g.quantile(.9),color=COLORS[i],alpha=.12)
        z=sub.groupby('dow').y.mean(); ax[1].plot(z.index,z,label=LABEL[t],color=COLORS[i])
    ax[0].set(xlabel='Fascia oraria',ylabel='Viaggi/ora osservata',title='Profilo orario e intervallo empirico 10–90%'); ax[1].set(title='Media per ora osservata, per giorno settimana',ylabel='Viaggi/ora osservata'); ax[1].set_xticks(range(7),['Lun','Mar','Mer','Gio','Ven','Sab','Dom']); ax[0].legend(); ax[1].legend(); save('04_shuttle_profiles.png')
    fig,ax=plt.subplots(1,2,figsize=(13,4.7))
    for i,t in enumerate(['DA APT','DA FS','SS']):
        z=sm[sm.type==t].set_index('month').reindex(sorted(sm.month.unique())); ax[0].plot(z.index,z.total,marker='o',label=LABEL[t],color=COLORS[i]); ax[1].plot(z.index,z.coverage_5_23_pct,marker='o',label=LABEL[t],color=COLORS[i])
    for a in ax: a.tick_params(axis='x',rotation=55); a.legend(fontsize=8)
    ax[0].set(title='Totali registrati nei file (copertura disomogenea)',ylabel='Viaggi registrati'); ax[1].set(title='Copertura delle fasce 5–23 nei mesi forniti',ylabel='% coppie giorno–ora presenti',ylim=(0,105)); save('05_shuttle_coverage.png')
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    allhours=pd.date_range(f.date.min(),f.date.max()+pd.Timedelta(hours=23),freq='h')
    days=['Lun','Mar','Mer','Gio','Ven','Sab','Dom']
    for i,(bound,typ) in enumerate([('A','DA APT'),('D','DA FS')]):
        series=v[v.BOUND==bound].groupby(v[v.BOUND==bound].BLOCK_TIME.dt.floor('h')).pax.sum().reindex(allhours,fill_value=0)
        temp=pd.DataFrame({'pax':series,'dow':allhours.dayofweek,'hour':allhours.hour}); grid=temp.pivot_table(index='dow',columns='hour',values='pax',aggfunc='mean').reindex(index=range(7),columns=range(24))
        im=ax[0,i].imshow(grid,aspect='auto',cmap='YlGnBu'); ax[0,i].set(title=('Arrivi' if bound=='A' else 'Partenze')+' aerei: pax/ora media',xlabel='Ora'); ax[0,i].set_xticks(range(0,24,3)); ax[0,i].set_yticks(range(7),days); ax[0,i].grid(False); fig.colorbar(im,ax=ax[0,i],shrink=.8)
        z=s[(s.type==typ)&s.hour.between(5,23)]; grid=z.pivot_table(index='dow',columns='hour',values='y',aggfunc='mean').reindex(index=range(7),columns=range(5,24))
        im=ax[1,i].imshow(grid,aspect='auto',cmap='YlGnBu'); ax[1,i].set(title=LABEL[typ]+': viaggi/ora osservata',xlabel='Ora'); ax[1,i].set_xticks(range(0,19,3),range(5,24,3)); ax[1,i].set_yticks(range(7),days); ax[1,i].grid(False); fig.colorbar(im,ax=ax[1,i],shrink=.8)
    save('09_weekly_heatmaps.png')
    return annual,sm

def flight_series(f,clock,pax='pax',extra=None):
    d=f[f.BOUND.isin(['A','D'])].copy()
    d['t']=d[clock].dt.floor('h')
    idx=pd.date_range(f.BLOCK_TIME.min().normalize()-pd.Timedelta(days=2),f.BLOCK_TIME.max().normalize()+pd.Timedelta(days=2),freq='h')
    series={}
    for b in ['A','D']:
        z=d[d.BOUND==b]
        series[b+'_pax']=z.groupby('t')[pax].sum().reindex(idx,fill_value=0)
        series[b+'_count']=z.groupby('t').size().reindex(idx,fill_value=0)
        if extra:
            for label,mask in extra.items(): series[b+'_'+label]=z[mask.loc[z.index]].groupby('t')[pax].sum().reindex(idx,fill_value=0)
    return series

def features(times,series):
    t=pd.DatetimeIndex(times); x=pd.DataFrame(index=range(len(t)))
    x['hour']=t.hour; x['dow']=t.dayofweek; x['month']=t.month; x['weekend']=(t.dayofweek>=5).astype(int)
    x['dayofyear_sin']=np.sin(2*np.pi*t.dayofyear/365.25); x['dayofyear_cos']=np.cos(2*np.pi*t.dayofyear/365.25)
    # Offset sign: A_-1 = arrivals one hour before target; D_+2 = departures two hours after.
    for key,z in series.items():
        offsets=range(-3,1) if key.startswith('A_') else range(0,5)
        for k in offsets: x[f'{key}_{k:+d}']=z.reindex(t+pd.Timedelta(hours=k)).to_numpy()
    return x

def relate(f,s):
    sr=flight_series(f,'BLOCK_TIME'); rows=[]
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for i,(typ,b) in enumerate([('DA APT','A'),('DA FS','D')]):
        z=s[(s.type==typ)&s.hour.between(5,23)].copy()
        z=z[z.time.between(f.BLOCK_TIME.min()+pd.Timedelta(days=1),f.BLOCK_TIME.max()-pd.Timedelta(days=1))]
        # Partial association after removing hour × weekday means, exploratory only.
        yr=z.y-z.groupby(['hour','dow']).y.transform('mean')
        for k in range(-4,5):
            x=sr[b+'_pax'].reindex(pd.DatetimeIndex(z.time)+pd.Timedelta(hours=k)).to_numpy()
            z['x']=x; xr=z.x-z.groupby(['hour','dow']).x.transform('mean')
            rows.append({'type':typ,'offset_hours':k,'n':len(z),'pearson_raw':z.y.corr(z.x),'pearson_residual':yr.corr(xr)})
        r=pd.DataFrame(rows); r=r[r.type==typ]
        ax[i].plot(r.offset_hours,r.pearson_raw,'o-',label='Correlazione grezza',color=COLORS[0]); ax[i].plot(r.offset_hours,r.pearson_residual,'o-',label='Dopo ora × giorno settimana',color=COLORS[1]); ax[i].axvline(0,color='gray',ls=':'); ax[i].set(title=LABEL[typ],xlabel='Ora volo − ora navetta',ylabel='Correlazione di Pearson',ylim=(-.4,.85)); ax[i].legend(fontsize=8)
    save('06_lag_correlations.png'); corr=pd.DataFrame(rows); corr.to_csv(OUT/'lag_correlations.csv',index=False)
    return corr

def estimator(name):
    if name=='Ridge calendario + pax stimati':
        cat=['hour','dow','month']; return make_pipeline(ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)],remainder=StandardScaler()),Ridge(alpha=100))
    return HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,max_depth=4,min_samples_leaf=35,l2_regularization=10,learning_rate=.06,early_stopping=False,random_state=42)

def metrics(y,p,threshold):
    y=np.asarray(y); p=np.asarray(p); peak=y>=threshold
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'WAPE_pct':100*np.abs(y-p).sum()/y.sum(),'R2':r2_score(y,p),'bias':float(np.mean(p-y)),'peak_MAE':mean_absolute_error(y[peak],p[peak]) if peak.any() else np.nan,'peak_bias':float(np.mean((p-y)[peak])) if peak.any() else np.nan,'n':len(y)}

def plot_model_errors(met):
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    names=['Media storica ora × settimana','Calendario','Calendario + voli programmati','Calendario + pax stimati','Retrospettivo pax e orari effettivi']
    for i,typ in enumerate(['DA APT','DA FS']):
        piv=met[(met.type==typ)&met.model.isin(names)].pivot(index='fold',columns='model',values='MAE').reindex(columns=names)
        piv.plot.bar(ax=ax[i],color=['#24374b','#8594a5',COLORS[0],COLORS[1],COLORS[2]]); ax[i].set(title=LABEL[typ],ylabel='MAE, viaggi/ora',xlabel='Mese di test'); ax[i].legend(fontsize=7); ax[i].tick_params(axis='x',rotation=30)
    save('07_model_errors.png')

def estimate_pax(f,cutoff):
    hist=f[(f.BLOCK_TIME<cutoff)&f.BOUND.isin(['A','D'])&(f.pax>0)].copy()
    # Passenger expectation is fitted only on flight records strictly before test start.
    means=hist.groupby(['FLIGHT_NUMBER','BOUND']).pax.agg(['mean','size']); reliable=means[means['size']>=5]['mean']
    key=pd.MultiIndex.from_frame(f[['FLIGHT_NUMBER','BOUND']]); est=pd.Series(reliable.reindex(key).to_numpy(),index=f.index)
    route=hist.groupby(['ORIDES_CODE','BOUND']).pax.mean(); fallback=route.reindex(pd.MultiIndex.from_frame(f[['ORIDES_CODE','BOUND']])).to_numpy()
    return est.fillna(pd.Series(fallback,index=f.index)).fillna(hist.pax.mean())

def model(f,s):
    valid=s[(s.type!='SS')&s.hour.between(5,23)&s.time.between(f.BLOCK_TIME.min()+pd.Timedelta(days=1),f.BLOCK_TIME.max()-pd.Timedelta(days=1))].copy()
    extra={'lowcost':f.LINEA_CHARTER.eq('2_LowCost'),'domestic':f.COUNTRY_NAME.eq('ITALY')}
    actual=flight_series(f,'BLOCK_TIME',extra=extra); sched=flight_series(f,'SCHEDULED_TIME')
    rows=[]; preds=[]; model_dir=OUT/'models'; model_dir.mkdir(exist_ok=True)
    folds=['2024-09','2025-03','2025-04','2025-05']
    for month in folds:
        cut=pd.Timestamp(month); end=cut+pd.offsets.MonthBegin(1)
        f['estimated_pax']=estimate_pax(f,cut); estimated=flight_series(f,'SCHEDULED_TIME','estimated_pax')
        if month=='2025-05':
            hist=f[(f.BLOCK_TIME<cut)&f.BOUND.isin(['A','D'])&(f.pax>0)]
            means=hist.groupby(['FLIGHT_NUMBER','BOUND']).pax.agg(['mean','size']); means=means[means['size']>=5].reset_index()
            routes=hist.groupby(['ORIDES_CODE','BOUND']).pax.mean().reset_index()
            (model_dir/'passenger_expectations.json').write_text(json.dumps({'cutoff':str(cut),'global_mean':float(hist.pax.mean()),'flights':means.to_dict('records'),'routes':routes.to_dict('records')},indent=2))
        for typ in ['DA APT','DA FS']:
            d=valid[valid.type==typ].reset_index(drop=True)
            train=d.time<cut; test=(d.time>=cut)&(d.time<end)
            if test.sum()<50 or train.sum()<200: continue
            log(f'Model fold {month} {typ}: train {train.sum()}, test {test.sum()}')
            xa=features(d.time,actual); xs=features(d.time,sched); xe=features(d.time,estimated)
            cal=[c for c in xa if not c.startswith(('A_','D_'))]
            count=cal+[c for c in xs if '_count_' in c]
            modes={'Calendario':xa[cal], 'Calendario + voli programmati':xs[count], 'Calendario + pax stimati':xe, 'Ridge calendario + pax stimati':xe, 'Retrospettivo pax e orari effettivi':xa}
            threshold=d.loc[train,'y'].quantile(.9)
            for name,x in modes.items():
                reg=estimator(name); reg.fit(x.loc[train],d.loc[train,'y']); p=np.maximum(0,reg.predict(x.loc[test]))
                m=metrics(d.loc[test,'y'],p,threshold); rows.append(dict(type=typ,fold=month,model=name,train_n=int(train.sum()),peak_threshold=float(threshold),**m))
                out=d.loc[test,['time','type','y']].copy(); out['prediction']=p; out['model']=name; out['fold']=month; out['peak_threshold']=threshold; preds.append(out)
                if month=='2025-05':
                    # Models are fitted through April; May predictions are genuinely held out.
                    import joblib
                    joblib.dump({'estimator':reg,'columns':list(x.columns),'cutoff':str(cut),'model':name,'type':typ},model_dir/f'{typ.replace(" ","_")}_{list(modes).index(name)}.joblib')
            # Grouped mean baseline, fitted strictly on earlier targets.
            group=d.loc[train].groupby(['hour','dow']).y.mean(); base=group.reindex(pd.MultiIndex.from_frame(d.loc[test,['hour','dow']])).to_numpy()
            base=np.where(np.isnan(base),d.loc[train,'y'].mean(),base)
            rows.append(dict(type=typ,fold=month,model='Media storica ora × settimana',train_n=int(train.sum()),peak_threshold=float(threshold),**metrics(d.loc[test,'y'],base,threshold)))
            out=d.loc[test,['time','type','y']].copy(); out['prediction']=base; out['model']='Media storica ora × settimana'; out['fold']=month; out['peak_threshold']=threshold; preds.append(out)
    met=pd.DataFrame(rows); pred=pd.concat(preds,ignore_index=True); met.to_csv(OUT/'model_metrics.csv',index=False); pred.to_csv(OUT/'predictions.csv',index=False)
    # Paired bootstrap by whole test day, not by correlated hourly rows.
    boot=[]; rng=np.random.default_rng(42)
    for typ in ['DA APT','DA FS']:
        for fold in folds:
            z=pred[(pred.type==typ)&(pred.fold==fold)].pivot(index='time',columns='model',values='prediction'); y=pred[(pred.type==typ)&(pred.fold==fold)].drop_duplicates('time').set_index('time').y.reindex(z.index)
            if z.empty: continue
            for reference in ['Calendario','Media storica ora × settimana']:
                for name in ['Calendario + voli programmati','Calendario + pax stimati','Retrospettivo pax e orari effettivi']:
                    err=pd.DataFrame({'base':np.abs(y-z[reference]),'gain':np.abs(y-z[reference])-np.abs(y-z[name])}); err['date']=err.index.normalize(); g=err.groupby('date').agg(gain=('gain','sum'),base=('base','sum'))
                    draws=rng.integers(0,len(g),size=(1000,len(g))); a=g.gain.to_numpy()[draws].sum(1)/g.base.to_numpy()[draws].sum(1)*100
                    boot.append({'type':typ,'fold':fold,'model':name,'reference':reference,'MAE_gain_pct':100*g.gain.sum()/g.base.sum(),'CI_low':np.quantile(a,.025),'CI_high':np.quantile(a,.975)})
    ci=pd.DataFrame(boot); ci.to_csv(OUT/'paired_bootstrap.csv',index=False)
    plot_model_errors(met)
    fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True)
    for i,typ in enumerate(['DA APT','DA FS']):
        z=pred[(pred.type==typ)&(pred.fold=='2025-05')]; obs=z.drop_duplicates('time').set_index('time').y
        # Fixed first seven days, avoiding cherry-picked examples.
        a=obs[obs.index<pd.Timestamp('2025-05-08')]; ax[i].plot(a.index,a,label='Osservato',color='#24374b',lw=1.4)
        for name,col in [('Calendario',COLORS[0]),('Calendario + pax stimati',COLORS[1])]:
            p=z[z.model==name].set_index('time').prediction; p=p[p.index<pd.Timestamp('2025-05-08')]; ax[i].plot(p.index,p,label=name,color=col,alpha=.8,lw=1)
        ax[i].set(title=LABEL[typ],ylabel='Viaggi/ora'); ax[i].legend(fontsize=8)
    save('08_predictions_week.png')
    # Post-hoc service stratification; SS never enters predictors.
    ss=s[s.type=='SS'].set_index('time').y; strata=[]
    for (typ,fold,name),z in pred.groupby(['type','fold','model']):
        z=z.copy(); z['ss']=ss.reindex(pd.DatetimeIndex(z.time)).to_numpy()
        for label,mask in [('SS=0',z.ss==0),('SS>0',z.ss>0),('SS assente',z.ss.isna())]:
            if mask.sum(): strata.append({'type':typ,'fold':fold,'model':name,'stratum':label,**metrics(z.loc[mask,'y'],z.loc[mask,'prediction'],z.peak_threshold.iloc[0])})
    pd.DataFrame(strata).to_csv(OUT/'service_strata.csv',index=False)
    return met,pred,ci,valid

def missing_and_ss(s):
    ss=s[s.type=='SS'].set_index('time').y; rows=[]
    for typ in ['DA APT','DA FS']:
        z=s[s.type==typ]; observed=pd.DatetimeIndex(z.time)
        for month in sorted(z.month.unique()):
            start=pd.Timestamp(month); dates=pd.date_range(start,start+pd.offsets.MonthEnd(0),freq='D'); idx=pd.DatetimeIndex([d+pd.Timedelta(hours=h) for d in dates for h in range(5,24)])
            missing=~idx.isin(observed); sy=ss.reindex(idx)
            for status,mask in [('presente',~missing),('assente',missing)]:
                v=sy.iloc[np.where(mask)[0]]
                rows.append({'type':typ,'month':month,'target_status':status,'hours':int(mask.sum()),'ss_hours_observed':int(v.notna().sum()),'ss_mean':v.mean(),'ss_positive_pct':100*(v.dropna()>0).mean()})
    r=pd.DataFrame(rows); r.to_csv(OUT/'missing_vs_ss.csv',index=False); return r

def zero_sensitivity(f,s):
    # Deliberately separate scenario, not a claim that absent records are zeros.
    results=[]
    for typ in ['DA APT','DA FS']:
        z=s[s.type==typ].set_index('time').y
        idx=[]
        for month in sorted(s[s.type==typ].month.unique()):
            start=pd.Timestamp(month)
            idx.extend(d+pd.Timedelta(hours=h) for d in pd.date_range(start,start+pd.offsets.MonthEnd(0),freq='D') for h in range(5,24))
        d=pd.DataFrame({'time':idx}); d['y']=z.reindex(pd.DatetimeIndex(d.time)).fillna(0).to_numpy()
        d=d[d.time.between(f.BLOCK_TIME.min()+pd.Timedelta(days=1),f.BLOCK_TIME.max()-pd.Timedelta(days=1))].reset_index(drop=True)
        cut=pd.Timestamp('2025-05'); train=d.time<cut; test=d.time.dt.strftime('%Y-%m')=='2025-05'
        f['estimated_pax']=estimate_pax(f,cut); x=features(d.time,flight_series(f,'SCHEDULED_TIME','estimated_pax')); cal=[c for c in x if not c.startswith(('A_','D_'))]
        for name,xx in [('Calendario',x[cal]),('Calendario + pax stimati',x)]:
            reg=estimator(name); reg.fit(xx.loc[train],d.loc[train,'y']); p=np.maximum(0,reg.predict(xx.loc[test])); results.append({'type':typ,'model':name,**metrics(d.loc[test,'y'],p,d.loc[train,'y'].quantile(.9))})
    r=pd.DataFrame(results); r.to_csv(OUT/'zero_sensitivity.csv',index=False); return r

def report(f,s,audit,annual,sm,corr,met,pred,ci,missing,valid,sensitivity):
    final=met[met.fold=='2025-05'].copy(); keycols=['type','model','MAE','RMSE','WAPE_pct','R2','peak_MAE','bias','n']
    common=f[f.BLOCK_TIME.dt.month<=6].groupby('year').agg(pax=('pax','sum'),records=('pax','size')).reset_index()
    # Ratios require full observed daytime coverage; no implied modal share.
    ratios=[]
    for typ,b in [('DA APT','A'),('DA FS','D')]:
        z=s[(s.type==typ)&s.hour.between(5,23)&s.date.between(f.date.min(),f.date.max())]; days=z.groupby('date').hour.nunique(); complete=days[days==19].index
        numerator=z[z.date.isin(complete)].y.sum(); denom=f[(f.BOUND==b)&f.date.isin(complete)].pax.sum()
        ratios.append({'direction':LABEL[typ],'complete_days_5_23':len(complete),'shuttle_daytime':numerator,'flight_daily_pax':denom,'aggregate_ratio_pct':100*numerator/denom})
    ratios=pd.DataFrame(ratios); ratios.to_csv(OUT/'aggregate_ratios.csv',index=False)
    peaks=s[s.type!='SS'].groupby('type').apply(lambda z:pd.Series({'mean_observed':z.y.mean(),'p95':z.y.quantile(.95),'p99':z.y.quantile(.99),'max':z.y.max(),'peak_hour_mean':int(z.groupby('hour').y.mean().idxmax())}),include_groups=False).reset_index()
    best=corr.loc[corr.groupby('type').pearson_residual.idxmax(),['type','offset_hours','pearson_raw','pearson_residual','n']]
    counts=f.groupby(['year','LINEA_CHARTER']).pax.sum().unstack(fill_value=0); shares=100*counts.div(counts.sum(axis=1),axis=0)
    lag=(f.BLOCK_TIME-f.SCHEDULED_TIME).dt.total_seconds()/60
    delay=f.assign(delta_min=lag).groupby('BOUND',dropna=False).delta_min.agg(['median',lambda x:x.quantile(.9)]).reset_index(); delay.columns=['BOUND','median_minutes','p90_minutes']
    high_missing=missing[(missing.target_status=='assente')&missing.ss_hours_observed.gt(0)]
    quality={'negative_pax':int((f.pax<0).sum()),'missing_pax':int(f.pax.isna().sum()),'transit_differences':int((f.PAX_ON_BOARD!=f.pax).sum()),'exact_flight_duplicates':int(f.drop(columns=['source','estimated_pax'],errors='ignore').duplicated().sum()),'delta_gt_24h':int((lag.abs()>1440).sum())}
    for typ in ['DA APT','DA FS']:
        z=final[final.type==typ]; base=z[z.model=='Calendario'].MAE.iloc[0]; op=z[z.model=='Calendario + pax stimati'].MAE.iloc[0]; log(f'Final {typ}: calendar MAE {base:.2f}, estimated pax MAE {op:.2f}, gain {100*(base-op)/base:.1f}%')
    findings=[]
    for typ in ['DA APT','DA FS']:
        z=final[final.type==typ].set_index('model'); op=z.loc['Calendario + pax stimati','MAE']; cal=z.loc['Calendario','MAE']; mean=z.loc['Media storica ora × settimana','MAE']
        findings.append(f"- **{LABEL[typ]}:** il modello con passeggeri stimati ha MAE {op:.1f} viaggi/ora, contro {cal:.1f} del calendario ({100*(cal-op)/cal:.1f}% di riduzione) e {mean:.1f} della media storica ora × settimana. Rispetto a quest'ultima, la variazione è {100*(mean-op)/mean:.1f}%: valori negativi indicano che la media semplice è migliore.")
    findings='\n'.join(findings)
    growth=100*(annual.loc[2024,'pax_no_transit']/annual.loc[2023,'pax_no_transit']-1)
    text=f'''# Traffico aereo di Bologna e utilizzo del Marconi Express

Analisi esplorativa e previsione dei viaggi passeggeri per fascia oraria. Elaborazione del 9 ottobre 2026.

## 1. Obiettivo e risultati principali

L'obiettivo è descrivere i flussi aerei e della navetta, misurare la loro relazione temporale e verificare se i voli migliorano una previsione oraria nelle due direzioni. Il risultato previsto è **il numero di viaggi passeggeri registrati per ora e direzione**, non l'occupazione di una singola corsa, la coda o la domanda non soddisfatta.

La verifica principale usa maggio 2025 come test finale, dopo addestramento sui mesi disponibili precedenti. I risultati su settembre 2024, marzo e aprile 2025 sono verifiche cronologiche ulteriori. Nessuna suddivisione casuale delle righe.

{findings}

Il legame con i voli è più convincente verso la stazione: il modello con passeggeri stimati supera entrambi i riferimenti in tutti e quattro i mesi. Verso l'aeroporto migliora sempre il calendario boosting, ma non sempre la semplice media storica. **Non emerge quindi un vantaggio universale del modello più complesso.** Nei picchi il modello a passeggeri stimati sottostima mediamente circa 63 viaggi/ora verso la stazione e 56 verso l'aeroporto nel test finale: le medie complessive non bastano per dimensionare il servizio.

{md(final[keycols])}

MAE e RMSE sono in viaggi/ora. WAPE è la somma degli errori assoluti divisa per i viaggi osservati. R² confronta con la media del test e può essere negativo. `peak_MAE` riguarda le ore il cui valore reale supera il 90° percentile dei dati di addestramento; `bias` positivo indica sovrastima. Il modello retrospettivo usa informazioni effettive non disponibili in anticipo e serve a valutare il potenziale informativo dei voli, non a dichiarare prestazioni operative.

Il contributo dei voli va letto confrontando ogni modello con entrambi i riferimenti, sulle stesse ore. Un miglioramento negativo indica peggioramento. Gli intervalli seguenti sono bootstrap appaiati per giornata (1.000 campioni), una misura empirica della variabilità nel mese, non una garanzia per stagioni future. La dipendenza fra giornate consecutive non è modellata dal bootstrap.

{md(ci[(ci.fold=='2025-05')&(ci.model=='Calendario + pax stimati')][['type','reference','MAE_gain_pct','CI_low','CI_high']])}

## 2. Fonti, unità e copertura

Fonte unica: [repository nbicocchi/paper-airport-express](https://github.com/nbicocchi/paper-airport-express), versione fissata dal commit `{json.loads((ROOT/'source_manifest.json').read_text())['sha']}`. Il file `source_manifest.json` riporta percorso, dimensione e SHA Git di ciascun file. Lo scaricamento verifica lo SHA, quindi il codice non dipende da modifiche successive del branch.

Sono letti tutti i **3 file aerei** e tutti i **29 file navetta** presenti in questa versione. Nessuna fonte esterna, nessun dato meteorologico o di servizio aggiunto.

### Dati aerei

{md(annual.reset_index())}

Il 2025 è parziale: confrontarne il totale con un anno completo sarebbe fuorviante. Il periodo comune gennaio–giugno consente questa lettura separata (verificare la copertura esatta nella tabella precedente):

{md(common)}

Una riga è un movimento. `BOUND=A` indica arrivo, `D` partenza. L'aeroporto di controparte è `ORIDES_CODE`. Si usa `PAX_ON_BOARD_NO_TRANSIT` per avvicinarsi ai flussi potenzialmente interessati al collegamento terrestre. `LINEA_CHARTER` distingue linea, low cost e charter; i codici di servizio non vengono decodificati senza legenda. Non si escludono movimenti con zero passeggeri dall'audit, mentre il loro contributo ai flussi passeggeri è nullo.

Gli orari sono trattati come tempi civili locali privi di timezone, assumendo che le due fonti condividano lo stesso riferimento e che la fascia h significhi [h:00, h+1:00). Questo va confermato: i file non specificano fuso, cambio ora né la semantica precisa di `BLOCK_TIME`. Quest'ultimo viene usato come riferimento effettivo al piazzale, non come istante di decollo/atterraggio.

### Dati della navetta

`DA APT`: aeroporto → stazione; `DA FS`: stazione → aeroporto; `SS`: sostitutivi. I sostitutivi **non hanno una colonna di direzione**, quindi non sono ripartiti artificialmente tra i due sensi. La colonna `viaggi sostitutive` è trattata come conteggio riportato dal file; per sommarla ai viaggi passeggeri ordinari occorre confermare che l'unità sia la stessa e che non vi sia sovrapposizione contabile.

{md(sm[['month','type','rows','days','total','coverage_5_23_pct']])}

La copertura percentuale usa come riferimento tecnico 19 fasce al giorno (5–23), non un orario ufficiale di esercizio. Le righe fuori da questa finestra sono mantenute nei totali descrittivi, ma escluse dalla previsione principale perché rare e con convenzioni poco chiare. Non vengono inseriti mesi non forniti. **Non esiste il file DA FS di agosto 2025.**

## 3. Qualità dei dati e limiti di osservazione

I file ordinari contengono solo valori positivi: **assenza di riga non equivale a zero certificato**. Mancano intere giornate in alcuni mesi; i totali sono somme registrate, non stime dei mesi interi. Nei file SS esistono righe con fascia `-`: vengono escluse dalle analisi orarie e il loro totale è riportato nell'audit. Nessuna riga è rimossa in modo silenzioso. Chiavi duplicate navetta data–ora–tipo provocano un errore nel codice.

Controlli aerei: {json.dumps(quality,ensure_ascii=False)}. I movimenti senza A/D sono mantenuti nei totali del file ma esclusi dai modelli direzionali. Duplicati esatti, se presenti, sono segnalati: senza identificatore univoco di movimento non viene applicata una deduplicazione arbitraria.

Le fasce ordinarie mancanti sono confrontate con SS quando il suo file è presente. Il dettaglio in `results/missing_vs_ss.csv` distingue presenza della misura ordinaria e presenza/positività di SS: serve a individuare possibili sostituzioni del servizio, non a provare che ogni assenza sia una sospensione.

Il segnale è forte in aprile e settembre 2024: in aprile, 227 delle 229 fasce mancanti DA APT hanno un valore SS disponibile e **tutte queste 227 sono positive**, con circa 72 viaggi sostitutivi medi; in settembre, tutte le 180 fasce mancanti con SS osservato sono positive, con circa 93 viaggi medi. In agosto 2024 SS è invece spesso positivo anche quando il servizio ordinario è registrato: può quindi rappresentare anche affiancamento, non solo sostituzione completa.

{md(high_missing[['type','month','hours','ss_hours_observed','ss_mean','ss_positive_pct']])}

I modelli principali imparano **condizionatamente alle fasce ordinarie osservate**. Se le assenze dipendono da sospensioni o sostituzioni, il campione non è rappresentativo di tutte le ore: questo è il principale limite operativo. SS non è usato come predittore perché il suo conteggio contemporaneo non sarebbe noto prima del viaggio; la stratificazione degli errori è solo retrospettiva (`service_strata.csv`).

## 4. Caratterizzazione del traffico aereo

![Movimenti e passeggeri mensili](figures/01_flights_monthly.png)

La lettura mensile separa volumi di movimenti e passeggeri. Gli ultimi mesi assenti del 2025 non sono zeri. Le serie descrivono stagionalità e crescita, ma tre anni di cui uno parziale non bastano per separare tutte le cause di cambiamento.

Nel 2024 i passeggeri esclusi transiti crescono del **{growth:.1f}%** rispetto al 2023. Il confronto è tra due anni interamente coperti; il 2025 termina invece il 10 luglio.

![Profilo orario dei voli](figures/02_flights_hourly.png)

L'ora effettiva identifica le concentrazioni giornaliere di passeggeri. Questi profili sono medie su tutti i giorni coperti, inclusi i giorni con zero movimenti in una determinata ora, quindi il denominatore è diverso da quello della navetta osservata.

![Paesi principali](figures/03_countries.png)

Quota di passeggeri per categoria commerciale, sul totale del file annuale:

{md(shares.reset_index())}

Differenza `BLOCK_TIME − SCHEDULED_TIME` per direzione (minuti):

{md(delay)}

Questa differenza è una misura tra i timestamp disponibili. Non viene chiamata ritardo di atterraggio/decollo senza confermare la definizione dei campi. Le code estreme e le date oltre il confine annuale richiedono attenzione.

## 5. Caratterizzazione del Marconi Express

![Profili navetta](figures/04_shuttle_profiles.png)

La banda mostra il 10°–90° percentile empirico dei conteggi per ora, non l'incertezza sulla media. Le medie per giorno settimana sono per **ora registrata**, quindi possono risentire delle differenze di copertura. I due sensi non devono essere sommati come se fossero un'unica domanda oraria.

{md(peaks)}

![Totali e copertura navetta](figures/05_shuttle_coverage.png)

Il grafico di copertura è necessario per interpretare quello dei volumi: una diminuzione del totale con meno ore osservate non prova una diminuzione della domanda. I sostitutivi sono mostrati separatamente; non si ricava un carico direzionale totale includendoli.

![Mappe settimana e ora per le due fonti](figures/09_weekly_heatmaps.png)

Le mappe consentono di confrontare le concentrazioni ora–giorno. I colori sono scalati separatamente per pannello: confrontare il ritmo, non l'intensità del colore tra dataset. Per i voli la media include tutte le ore dei giorni coperti; per la navetta include solo le righe osservate. Differenze di campionamento e composizione dei mesi possono influire sui profili settimanali.

## 6. Relazione temporale fra le fonti

L'abbinamento avviene su data e ora, non su singolo passeggero. Le finestre considerate sono:

- aeroporto → stazione: arrivi nelle ore t−3, t−2, t−1, t;
- stazione → aeroporto: partenze nelle ore t, t+1, t+2, t+3, t+4.

Entrambi i modelli ricevono entrambe le finestre, per consentire effetti incrociati. L'anticipo delle partenze rappresenta il tempo con cui il viaggiatore raggiunge l'aeroporto prima del volo; non è fuga di informazioni se riguarda il programma disponibile in anticipo. Diventa invece informazione futura quando si usano orari/passeggeri effettivi, come nel modello retrospettivo.

![Correlazioni con spostamento temporale](figures/06_lag_correlations.png)

Il segno dell'asse è **ora del volo meno ora della navetta**. Valori negativi: voli prima del viaggio; positivi: voli dopo. Sono mostrate anche correlazioni dopo sottrazione della media ora × giorno settimana, per ridurre l'effetto del ritmo giornaliero comune. È una relazione esplorativa, non un effetto causale, e la fascia migliore non viene selezionata sul test per cambiare il modello.

Massimi della correlazione depurata nelle finestre esplorate:

{md(best)}

Verso la stazione il segnale si concentra nella stessa ora e nell'ora precedente (correlazioni depurate circa 0,38). Verso l'aeroporto il massimo è con le partenze tre ore dopo (circa 0,38), con un segnale anche due ore dopo. La risoluzione oraria non consente di dedurre un tempo individuale preciso di trasferimento.

Rapporto aggregato su giornate con tutte le fasce ordinarie 5–23 presenti:

{md(ratios)}

Questo rapporto confronta viaggi navetta diurni con passeggeri aerei dell'intero giorno nella direzione corrispondente. **Non è una quota modale individuale**: non osserviamo persone, collegamenti ferroviari, altri mezzi, personale aeroportuale o lo spostamento tra giorni. Le giornate complete selezionano inoltre un sottocampione.

## 7. Modelli e verifica su dati successivi

### Modelli confrontati

1. Media storica ora × giorno settimana: riferimento semplice, stimato solo sul passato.
2. Calendario: gradient boosting su ora, giorno settimana, mese, weekend e seno/coseno del giorno dell'anno.
3. Calendario + voli programmati: aggiunge conteggi dei voli nelle finestre precedenti/successive, senza passeggeri effettivi.
4. Calendario + pax stimati: usa il programma storico e una stima dei passeggeri per numero volo e direzione, calcolata **solo dai movimenti precedenti al mese di test**. Sono richieste almeno cinque osservazioni storiche; fallback per aeroporto e direzione, poi media globale storica.
5. Ridge calendario + pax stimati: riferimento lineare regolarizzato, con variabili categoriali codificate e variabili numeriche standardizzate sul solo training.
6. Retrospettivo pax e orari effettivi: usa passeggeri reali, `BLOCK_TIME` e composizione low cost/domestica. È un benchmark informativo a posteriori.

Gradient boosting: 220 iterazioni, learning rate 0,06, massimo 15 foglie e profondità 4, minimo 35 campioni/foglia, regolarizzazione L2=10. Parametri fissati, nessuna ricerca sul test, nessun early stopping casuale. Ridge alpha=100. Le predizioni vengono troncate a zero. Non vengono usati conteggi navetta futuri o contemporanei come input.

### Suddivisione temporale

Per ciascun mese di test si addestra su tutte le righe ordinarie disponibili strettamente precedenti al primo giorno del mese. Non si interpolano i mesi mancanti. Il 2023 aiuta la stima storica dei passeggeri aerei, ma non contiene target navetta. I mesi navetta successivi al termine dei dati aerei vengono esclusi dal collegamento e dai modelli, mantenuti nella descrizione.

{md(met[['type','fold','model','train_n','n','MAE','WAPE_pct','R2','peak_MAE']])}

![Errori fuori campione](figures/07_model_errors.png)

La variabilità tra i mesi è parte del risultato: un buon modello deve migliorare il calendario con una certa stabilità e contenere l'errore nelle punte. Non si sceglie un vincitore solo perché ha il minimo errore su un singolo mese.

![Predizioni prima settimana di maggio 2025](figures/08_predictions_week.png)

Il periodo illustrato è prefissato: 1–7 maggio, non una settimana scelta per mostrare prestazioni favorevoli. Le ore non registrate non sono imputate nel modello principale.

### Sensibilità all'ipotesi «assenza = zero»

Ripetizione separata su maggio 2025 dopo aver inserito zeri in tutte le fasce mancanti 5–23 dei mesi forniti:

{md(sensitivity[['type','model','MAE','WAPE_pct','R2','n']])}

È uno scenario ipotetico, **non la correzione del dataset**. Se le prestazioni cambiano, significa che il significato delle assenze è determinante. Non adottare questo scenario per pianificare il servizio senza validarlo con il gestore.

## 8. Utilizzo operativo e sviluppi

Le prestazioni del modello a passeggeri stimati sono una simulazione più vicina alla previsione anticipata rispetto al benchmark effettivo, ma il programma contenuto nei file è retrospettivo: non abbiamo snapshot del programma conosciuto al momento della previsione. Non è ancora una validazione prospettica.

Per trasformare l'analisi in uno strumento di previsione:

- confermare timezone, fascia oraria, conteggi SS, assenze, definizione BLOCK_TIME e date di effettiva copertura;
- acquisire serie navetta continue con zeri espliciti, stato servizio e direzione dei sostitutivi;
- usare il programma dei voli disponibile al momento della previsione, con aggiornamenti, cancellazioni, capacità e stime del riempimento;
- prevedere separatamente stato di esercizio e viaggi passeggeri, poi valutare su tutti i giorni in finestre cronologiche mobili;
- aggiungere festività, eventi, meteo, tariffe e collegamenti ferroviari quando disponibili;
- associare i conteggi alle corse, capacità e frequenze per stimare saturazione e code: i file attuali non permettono quel passaggio;
- produrre intervalli di previsione calibrati su periodi futuri. Le bande nei profili e il bootstrap sul guadagno non sono intervalli individuali di previsione.

## 9. Riproducibilità e contenuti

`download.py` scarica i file dalla versione fissata e ne verifica gli SHA. `analyze.py` esegue pulizia, descrizione, grafici, correlazioni, stima passeggeri, modelli, bootstrap e genera questo documento. `requirements.txt` specifica le versioni usate. `results/` contiene audit, tabelle, predizioni fuori campione e modelli del test maggio (addestrati fino ad aprile). I modelli salvati non sono riaddestrati sul test e non sono un servizio di produzione. Caricare file joblib solo da fonti fidate.

`predict.py` applica i due modelli con passeggeri stimati a un CSV di voli programmati e a un CSV di ore richieste. Il primo richiede `SCHEDULED_TIME, FLIGHT_NUMBER, BOUND, ORIDES_CODE`; il secondo `time, type`, con `type` pari a `DA APT` o `DA FS`. Occorre fornire **il programma completo da tre ore prima a quattro ore dopo le ore richieste**: voli assenti dal CSV vengono interpretati come zero voli. Nessun passeggero effettivo del futuro entra nella stima. Esempi e risultati di verifica sono in `examples/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download.py
python analyze.py
python predict.py --schedule examples/schedule_may.csv --targets examples/targets_may.csv --output examples/predicted_may.csv
```

Eseguire dalla cartella `code/` del repository (il codice risolve i percorsi in base alla propria posizione). Il `source_manifest.json` deve trovarsi accanto al codice. I file sorgenti si trovano in `../dataset/` e sono verificati prima della lettura; il download serve solo a ripristinare file mancanti. Python utilizzato: {platform.python_version()}; pandas {pd.__version__}; scikit-learn {sklearn.__version__}; numpy {np.__version__}. Random seed 42. Non viene scritto alcun Excel né modificato il repository remoto.
'''
    (ROOT/'report.md').write_text(text,encoding='utf-8')
    (ROOT/'requirements.txt').write_text('\n'.join(f'{name}=={__import__(name).__version__}' for name in ['numpy','pandas','matplotlib','sklearn','openpyxl','scipy','joblib']).replace('sklearn==','scikit-learn==')+'\n')

if __name__=='__main__':
    f,s,audit=load(); annual,sm=describe(f,s); corr=relate(f,s); missing=missing_and_ss(s)
    met,pred,ci,valid=model(f,s); sensitivity=zero_sensitivity(f,s)
    report(f,s,audit,annual,sm,corr,met,pred,ci,missing,valid,sensitivity)
    log('Report and artifacts complete')
