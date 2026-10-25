"""Read immutable Excel inputs, summarize coverage, and run the holdout benchmark."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_sources import DATA_ROOT, verify_sources

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
OUT.mkdir(exist_ok=True)


def md(frame):
    def fmt(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, (float, np.floating)):
            return f'{value:.2f}'
        return str(value).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |'] + ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])


def load():
    verify_sources()
    flights, shuttle, audit = [], [], []
    for path in sorted((DATA_ROOT/'Aeroporto').glob('*.xlsx')):
        d = pd.read_excel(path)
        d['source'] = path.name
        for column in ['SCHEDULED_TIME', 'BLOCK_TIME']:
            d[column] = pd.to_datetime(d[column], errors='coerce')
        for column in ['PAX_ON_BOARD', 'PAX_ON_BOARD_NO_TRANSIT', 'BAG_NUMBER']:
            d[column] = pd.to_numeric(d[column], errors='coerce')
        audit.append(dict(file=path.name, rows=len(d), first=str(d.BLOCK_TIME.min()), last=str(d.BLOCK_TIME.max()), missing_bound=int((~d.BOUND.isin(['A', 'D'])).sum()), duplicates_exact=int(d.drop(columns='source').duplicated().sum())))
        flights.append(d)
    f = pd.concat(flights, ignore_index=True)
    for path in sorted((DATA_ROOT/'Marconi Express').glob('*.xlsx')):
        raw = pd.read_excel(path)
        d = raw.iloc[:, :3].copy()
        d.columns = ['date', 'hour', 'y']
        d['date'] = pd.to_datetime(d.date, errors='coerce')
        d['hour'] = pd.to_numeric(d.hour, errors='coerce')
        d['y'] = pd.to_numeric(d.y, errors='coerce')
        valid = d.date.notna() & d.hour.between(0, 23) & (d.hour % 1 == 0) & d.y.notna() & (d.y >= 0)
        clean = d[valid].copy()
        clean['type'] = 'SS' if path.name.startswith('SS') else ('DA APT' if path.name.startswith('DA APT') else 'DA FS')
        clean['source'] = path.name
        clean['time'] = clean.date+pd.to_timedelta(clean.hour, unit='h')
        clean['month'] = clean.date.dt.strftime('%Y-%m')
        audit.append(dict(file=path.name, rows=len(raw), valid_rows=len(clean), excluded_rows=int((~valid).sum()), excluded_y=float(d.loc[~valid, 'y'].sum()), first=str(clean.date.min()), last=str(clean.date.max()), days=clean.date.nunique(), zeros=int((clean.y == 0).sum()), duplicates_exact=int(clean.duplicated(['time']).sum())))
        shuttle.append(clean)
    s = pd.concat(shuttle, ignore_index=True).sort_values('time')
    assert not s.duplicated(['type', 'time']).any(), 'duplicate shuttle keys'
    # Exact duplicate flights are reported, not dropped without a movement ID.
    f['pax'] = f.PAX_ON_BOARD_NO_TRANSIT
    f['date'] = f.BLOCK_TIME.dt.normalize()
    f['month'] = f.BLOCK_TIME.dt.strftime('%Y-%m')
    f['hour'], f['year'] = f.BLOCK_TIME.dt.hour, f.BLOCK_TIME.dt.year
    s['dow'] = s.date.dt.dayofweek
    pd.DataFrame(audit).to_csv(OUT/'source_audit.csv', index=False)
    s.to_csv(OUT/'shuttle_clean.csv', index=False)
    print(f'Loaded {len(f)} flights, {len(s)} valid shuttle rows', flush=True)
    return f, s, pd.DataFrame(audit)


def describe(f, s):
    annual = f.groupby('year').agg(records=('pax', 'size'), pax_no_transit=('pax', 'sum'), days=('date', 'nunique'), first=('date', 'min'), last=('date', 'max'))
    annual['arrivals'] = f[f.BOUND == 'A'].groupby('year').size()
    annual['departures'] = f[f.BOUND == 'D'].groupby('year').size()
    annual['unclassified'] = f[~f.BOUND.isin(['A', 'D'])].groupby('year').size()
    annual.fillna({'unclassified': 0}).reset_index().to_csv(OUT/'flights_annual.csv', index=False)
    f.groupby(['month', 'BOUND'], dropna=False).agg(movements=('pax', 'size'), pax=('pax', 'sum')).reset_index().to_csv(OUT/'flights_monthly.csv', index=False)
    monthly = s.groupby(['month', 'type']).agg(rows=('y', 'size'), days=('date', 'nunique'), total=('y', 'sum'), mean_observed=('y', 'mean'), p95=('y', lambda x: x.quantile(.95)), max=('y', 'max')).reset_index()
    monthly['hours_5_23'] = monthly.apply(lambda r: int(((s.month == r.month) & (s.type == r.type) & s.hour.between(5, 23)).sum()), axis=1)
    monthly['expected_5_23'] = pd.to_datetime(monthly.month).dt.days_in_month*19
    monthly['coverage_5_23_pct'] = 100*monthly.hours_5_23/monthly.expected_5_23
    monthly.to_csv(OUT/'shuttle_monthly.csv', index=False)


def flight_series(f, clock, pax='pax'):
    d = f[f.BOUND.isin(['A', 'D'])].copy()
    d['t'] = d[clock].dt.floor('h')
    idx = pd.date_range(f.BLOCK_TIME.min().normalize()-pd.Timedelta(days=2), f.BLOCK_TIME.max().normalize()+pd.Timedelta(days=2), freq='h')
    series = {}
    for bound in ['A', 'D']:
        z = d[d.BOUND == bound]
        series[bound+'_pax'] = z.groupby('t')[pax].sum().reindex(idx, fill_value=0)
        series[bound+'_count'] = z.groupby('t').size().reindex(idx, fill_value=0)
    return series


def features(times, series):
    t = pd.DatetimeIndex(times)
    x = pd.DataFrame(index=range(len(t)))
    x['hour'], x['dow'], x['month'] = t.hour, t.dayofweek, t.month
    x['weekend'] = (t.dayofweek >= 5).astype(int)
    x['dayofyear_sin'] = np.sin(2*np.pi*t.dayofyear/365.25)
    x['dayofyear_cos'] = np.cos(2*np.pi*t.dayofyear/365.25)
    # Arrival offsets -3..0; departure offsets 0..4, relative to target hour.
    for key, z in series.items():
        offsets = range(-3, 1) if key.startswith('A_') else range(0, 5)
        for offset in offsets:
            x[f'{key}_{offset:+d}'] = z.reindex(t+pd.Timedelta(hours=offset)).to_numpy()
    return x


def relate(f, s):
    series = flight_series(f, 'BLOCK_TIME')
    rows = []
    for typ, bound in [('DA APT', 'A'), ('DA FS', 'D')]:
        z = s[(s.type == typ) & s.hour.between(5, 23)].copy()
        z = z[z.time.between(f.BLOCK_TIME.min()+pd.Timedelta(days=1), f.BLOCK_TIME.max()-pd.Timedelta(days=1))]
        yr = z.y-z.groupby(['hour', 'dow']).y.transform('mean')
        for offset in range(-4, 5):
            z['x'] = series[bound+'_pax'].reindex(pd.DatetimeIndex(z.time)+pd.Timedelta(hours=offset)).to_numpy()
            xr = z.x-z.groupby(['hour', 'dow']).x.transform('mean')
            rows.append(dict(type=typ, offset_hours=offset, n=len(z), pearson_raw=z.y.corr(z.x), pearson_residual=yr.corr(xr)))
    pd.DataFrame(rows).to_csv(OUT/'lag_correlations.csv', index=False)


def estimator(name):
    return HistGradientBoostingRegressor(max_iter=220, max_leaf_nodes=15, max_depth=4, min_samples_leaf=35, l2_regularization=10, learning_rate=.06, early_stopping=False, random_state=42)


def metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    peak = y >= threshold
    return dict(MAE=mean_absolute_error(y, p), RMSE=np.sqrt(mean_squared_error(y, p)), WAPE_pct=100*np.abs(y-p).sum()/y.sum(), R2=r2_score(y, p), bias=float(np.mean(p-y)), peak_MAE=mean_absolute_error(y[peak], p[peak]) if peak.any() else np.nan, peak_bias=float(np.mean((p-y)[peak])) if peak.any() else np.nan, n=len(y))


def estimate_pax(f, cutoff):
    hist = f[(f.BLOCK_TIME < cutoff) & f.BOUND.isin(['A', 'D']) & (f.pax > 0)]
    means = hist.groupby(['FLIGHT_NUMBER', 'BOUND']).pax.agg(['mean', 'size'])
    reliable = means[means['size'] >= 5]['mean']
    key = pd.MultiIndex.from_frame(f[['FLIGHT_NUMBER', 'BOUND']])
    estimate = pd.Series(reliable.reindex(key).to_numpy(), index=f.index)
    route = hist.groupby(['ORIDES_CODE', 'BOUND']).pax.mean()
    fallback = route.reindex(pd.MultiIndex.from_frame(f[['ORIDES_CODE', 'BOUND']])).to_numpy()
    return estimate.fillna(pd.Series(fallback, index=f.index)).fillna(hist.pax.mean())


def missing_and_ss(s):
    ss = s[s.type == 'SS'].set_index('time').y
    rows = []
    for typ in ['DA APT', 'DA FS']:
        z = s[s.type == typ]
        for month in sorted(z.month.unique()):
            start = pd.Timestamp(month)
            idx = pd.date_range(start, start+pd.offsets.MonthBegin(1), freq='h', inclusive='left')
            idx = idx[idx.hour.isin(range(5, 24))]
            missing = ~idx.isin(z.time)
            sy = ss.reindex(idx)
            for status, mask in [('presente', ~missing), ('assente', missing)]:
                v = sy.iloc[np.where(mask)[0]]
                rows.append(dict(type=typ, month=month, target_status=status, hours=int(mask.sum()), ss_hours_observed=int(v.notna().sum()), ss_mean=v.mean(), ss_positive_pct=100*(v.dropna() > 0).mean()))
    pd.DataFrame(rows).to_csv(OUT/'missing_vs_ss.csv', index=False)


if __name__ == '__main__':
    # Import after module initialization so revision.py can reuse the helpers.
    from revision import run
    flights, shuttle, _ = load()
    describe(flights, shuttle)
    relate(flights, shuttle)
    missing_and_ss(shuttle)
    run(flights, shuttle)
    print('Descriptive results and single-split experiments complete.', flush=True)
