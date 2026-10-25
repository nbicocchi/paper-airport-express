"""Example inference from a scheduled-flight CSV, using the May 2025 test models.
These models are frozen through April 2025, not a production forecast service.
"""
from pathlib import Path
import argparse, json, os
os.environ.setdefault('OMP_NUM_THREADS','2')
import pandas as pd
import numpy as np
import joblib
from analyze import features

ROOT=Path(__file__).resolve().parent
def predict(schedule,targets,model_dir=ROOT/'results'/'models'):
    required={'SCHEDULED_TIME','FLIGHT_NUMBER','BOUND','ORIDES_CODE'}
    if not required.issubset(schedule): raise ValueError(f'Required columns: {sorted(required)}')
    d=schedule.copy(); d['t']=pd.to_datetime(d.SCHEDULED_TIME,errors='raise').dt.floor('h')
    if not d.BOUND.isin(['A','D']).all(): raise ValueError('BOUND must be A or D')
    t=pd.DatetimeIndex(pd.to_datetime(targets.time,errors='raise'))
    if not targets.type.isin(['DA APT','DA FS']).all(): raise ValueError('type must be DA APT or DA FS')
    if not t.hour.isin(range(5,24)).all(): raise ValueError('Models evaluated only for hours 5–23')
    meta=json.loads((model_dir/'passenger_expectations.json').read_text())
    means=pd.DataFrame(meta['flights']).set_index(['FLIGHT_NUMBER','BOUND'])['mean']
    routes=pd.DataFrame(meta['routes']).set_index(['ORIDES_CODE','BOUND']).pax
    d['pax']=means.reindex(pd.MultiIndex.from_frame(d[['FLIGHT_NUMBER','BOUND']])).to_numpy()
    fallback=routes.reindex(pd.MultiIndex.from_frame(d[['ORIDES_CODE','BOUND']])).to_numpy()
    d['pax']=d.pax.fillna(pd.Series(fallback,index=d.index)).fillna(meta['global_mean'])
    idx=pd.date_range(t.min()-pd.Timedelta(hours=3),t.max()+pd.Timedelta(hours=4),freq='h')
    series={}
    for b in ['A','D']:
        z=d[d.BOUND==b]; series[b+'_pax']=z.groupby('t').pax.sum().reindex(idx,fill_value=0); series[b+'_count']=z.groupby('t').size().reindex(idx,fill_value=0)
    x=features(t,series); out=targets.copy(); out['prediction']=np.nan
    for typ in ['DA APT','DA FS']:
        mask=targets.type.eq(typ).to_numpy()
        if mask.any():
            bundle=joblib.load(model_dir/f'{typ.replace(" ","_")}_2.joblib')
            out.loc[mask,'prediction']=np.maximum(0,bundle['estimator'].predict(x.loc[mask,bundle['columns']]))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--schedule',required=True); ap.add_argument('--targets',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    predict(pd.read_csv(args.schedule),pd.read_csv(args.targets)).to_csv(args.output,index=False)
    print('Saved predictions. Assumes complete schedule for target hours −3 to +4; absent flights are treated as zero. Confirm availability and service state before operational use.')
