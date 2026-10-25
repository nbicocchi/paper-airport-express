"""Single-split experiments: fit before March 2025, test March--May once.

Run directly, or via analyze.py to refresh descriptive results as well.
No archived forecast-issuance times are assumed. Only RECENT updates in testing.
"""
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from analyze import load, features, flight_series, estimate_pax, estimator, metrics, md
from revision_io import compress_outputs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'revision'
CUTOFF = pd.Timestamp('2025-03-01')
TEST_END = pd.Timestamp('2025-06-01')
FOLDS = ['2025-03', '2025-04', '2025-05']
MAIN_MODELS = ['CAL', 'SCHED', 'PAX']
DIAGNOSTICS = ['REAL-LOAD', 'REAL-TIME', 'REAL-BOTH']
MODELS = MAIN_MODELS + ['MEAN', 'RECENT'] + DIAGNOSTICS


def periods(frame):
    """Aggregate the same held-out predictions, without duplicating saved rows."""
    yield 'ALL', frame
    for fold, rows in frame.groupby('fold', sort=True):
        yield fold, rows


def expectation_level(f):
    h = f[(f.BLOCK_TIME < CUTOFF) & f.BOUND.isin(['A', 'D']) & (f.pax > 0)]
    n = h.groupby(['FLIGHT_NUMBER', 'BOUND']).size()
    flight = pd.MultiIndex.from_frame(f[['FLIGHT_NUMBER', 'BOUND']]).isin(n[n >= 5].index)
    route = pd.MultiIndex.from_frame(f[['ORIDES_CODE', 'BOUND']]).isin(h.groupby(['ORIDES_CODE', 'BOUND']).size().index)
    return np.where(flight, 'flight', np.where(route, 'airport', 'global'))


def paired(pred):
    rows = []
    rng = np.random.default_rng(42)
    for fold, panel in periods(pred):
        for typ, z in panel.groupby('type'):
            p = z.pivot(index='time', columns='model', values='prediction')
            y = z.drop_duplicates('time').set_index('time').y.reindex(p.index)
            for ref in ['CAL', 'SCHED', 'MEAN', 'RECENT', 'PAX']:
                for model in (DIAGNOSTICS if ref == 'PAX' else ['SCHED', 'PAX']):
                    if model == ref:
                        continue
                    daily = pd.DataFrame({'base': abs(y-p[ref]), 'model': abs(y-p[model])}).groupby(p.index.normalize()).sum()
                    draws = rng.integers(0, len(daily), size=(1000, len(daily)))
                    b, m = daily.base.to_numpy(), daily.model.to_numpy()
                    gain = 100*(1-m[draws].sum(1)/b[draws].sum(1))
                    rows.append(dict(type=typ, fold=fold, model=model, reference=ref,
                                     gain_pct=100*(1-m.sum()/b.sum()), low=np.quantile(gain, .025), high=np.quantile(gain, .975), days=len(daily), hours=len(y)))
    return pd.DataFrame(rows)


def directional(pred):
    rows = []
    rng = np.random.default_rng(43)
    for fold, z in periods(pred):
        panels = {}
        for typ in ['DA APT', 'DA FS']:
            d = z[z.type == typ]
            p = d.pivot(index='time', columns='model', values='prediction')
            panels[typ] = (p, d.drop_duplicates('time').set_index('time').y.reindex(p.index))
        common = panels['DA APT'][0].index.intersection(panels['DA FS'][0].index)
        for ref in ['CAL', 'SCHED', 'MEAN', 'RECENT']:
            daily = []
            for typ in ['DA APT', 'DA FS']:
                p, y = panels[typ]
                p, y = p.reindex(common), y.reindex(common)
                daily.append(pd.DataFrame({'b': abs(y-p[ref]), 'm': abs(y-p.PAX)}).groupby(common.normalize()).sum())
            draws = rng.integers(0, len(daily[0]), size=(1000, len(daily[0])))
            gains = [100*(1-d.m.sum()/d.b.sum()) for d in daily]
            samples = [100*(1-d.m.to_numpy()[draws].sum(1)/d.b.to_numpy()[draws].sum(1)) for d in daily]
            diff = samples[0]-samples[1]
            rows.append(dict(fold=fold, reference=ref, apt_gain=gains[0], fs_gain=gains[1], difference_pp=gains[0]-gains[1], low=np.quantile(diff, .025), high=np.quantile(diff, .975), common_hours=len(common), common_days=len(daily[0])))
    return pd.DataFrame(rows)


def summarize(pred, qpred, train_counts, edges):
    scores, peaks, calibration, quantiles = [], [], [], []
    for fold, panel in periods(pred):
        for (typ, model), z in panel.groupby(['type', 'model']):
            threshold = z.peak_threshold.iloc[0]
            scores.append(dict(type=typ, fold=fold, model=model, train_n=train_counts[typ], peak_threshold=threshold, **metrics(z.y, z.prediction, threshold)))
            e = (z.prediction-z.y)[z.y >= threshold]
            peaks.append(dict(type=typ, fold=fold, model=model, peak_n=len(e), n=len(z), peak_bias=e.mean(), peak_MAE=abs(e).mean(), under_pct=100*(e < 0).mean(), error_p10=e.quantile(.1), error_p50=e.quantile(.5), error_p90=e.quantile(.9)))
        for typ, z in panel[panel.model == 'PAX'].groupby('type'):
            bins = pd.cut(z.prediction, edges[typ], labels=False, include_lowest=True)
            for group, d in z.groupby(bins):
                calibration.append(dict(type=typ, fold=fold, bin=int(group), n=len(d), predicted_mean=d.prediction.mean(), observed_mean=d.y.mean(), bias=(d.prediction-d.y).mean()))
            q = qpred[qpred.type == typ]
            if fold != 'ALL':
                q = q[q.fold == fold]
            q = q.set_index('time').reindex(z.time)
            y = z.y.to_numpy()
            peak = y >= z.peak_threshold.iloc[0]
            quantiles.append(dict(type=typ, fold=fold, n=len(y), q90_coverage=100*np.mean(y <= q.q90), peak_n=int(peak.sum()), peak_coverage=100*np.mean(y[peak] <= q.q90.to_numpy()[peak]), q90_pinball=mean_pinball_loss(y, q.q90, alpha=.9), point_pinball=mean_pinball_loss(y, z.prediction, alpha=.9)))
    for name, rows in [('model_metrics', scores), ('peak_diagnostics', peaks), ('calibration', calibration), ('quantile_metrics', quantiles)]:
        pd.DataFrame(rows).to_csv(OUT / (name+'.csv'), index=False)
    return pd.DataFrame(scores)


def zero_sensitivity(f, s):
    """Separate scenario: zero-fill missing hours only within supplied months."""
    rows = []
    for typ in ['DA APT', 'DA FS']:
        observed = s[s.type == typ]
        grids = []
        for month in sorted(observed.month.unique()):
            start = pd.Timestamp(month)
            if start >= TEST_END:
                continue
            grid = pd.date_range(start, start+pd.offsets.MonthBegin(1), freq='h', inclusive='left')
            grids.append(grid[grid.hour.isin(range(5, 24))])
        times = grids[0].append(grids[1:])
        d = pd.DataFrame({'time': times})
        d['y'] = observed.set_index('time').y.reindex(times).fillna(0).to_numpy()
        tr, te = d.time < CUTOFF, d.time.between(CUTOFF, TEST_END, inclusive='left')
        x = features(d.time, flight_series(f, 'SCHEDULED_TIME', 'estimated_pax'))
        columns = {'CAL': [c for c in x if not c.startswith(('A_', 'D_'))], 'SCHED': [c for c in x if '_pax_' not in c], 'PAX': list(x)}
        for name, cols in columns.items():
            reg = estimator(name)
            reg.fit(x.loc[tr, cols], d.loc[tr, 'y'])
            out = d.loc[te].copy()
            out['prediction'] = np.maximum(0, reg.predict(x.loc[te, cols]))
            out['fold'] = out.time.dt.strftime('%Y-%m')
            for fold, z in periods(out):
                rows.append(dict(type=typ, fold=fold, model=name, train_n=int(tr.sum()), **metrics(z.y, z.prediction, d.loc[tr, 'y'].quantile(.9))))
    pd.DataFrame(rows).to_csv(ROOT/'results'/'zero_sensitivity.csv', index=False)


def run(f, s):
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT/'results'/'models'
    model_dir.mkdir(exist_ok=True)
    f = f.copy()
    f['estimated_pax'] = estimate_pax(f, CUTOFF)
    level = expectation_level(f)
    hist = f[(f.BLOCK_TIME < CUTOFF) & f.BOUND.isin(['A', 'D']) & (f.pax > 0)]
    means = hist.groupby(['FLIGHT_NUMBER', 'BOUND']).pax.agg(['mean', 'size'])
    routes = hist.groupby(['ORIDES_CODE', 'BOUND']).pax.mean().reset_index()
    (model_dir/'passenger_expectations.json').write_text(json.dumps({'cutoff': str(CUTOFF), 'global_mean': float(hist.pax.mean()), 'flights': means[means['size'] >= 5].reset_index().to_dict('records'), 'routes': routes.to_dict('records')}, indent=2)+'\n')
    series = {name: flight_series(f, clock, pax) for name, clock, pax in [
        ('PAX', 'SCHEDULED_TIME', 'estimated_pax'), ('REAL-LOAD', 'SCHEDULED_TIME', 'pax'),
        ('REAL-TIME', 'BLOCK_TIME', 'estimated_pax'), ('REAL-BOTH', 'BLOCK_TIME', 'pax')]}
    valid = s[(s.type != 'SS') & s.hour.between(5, 23) & s.time.between(f.BLOCK_TIME.min()+pd.Timedelta(days=1), f.BLOCK_TIME.max()-pd.Timedelta(days=1)) & (s.time < TEST_END)].copy()
    preds, qpreds, audit, fallback, samples, fit_audit = [], [], [], [], [], []
    train_counts, edges = {}, {}
    for fold, start, end in [('ALL', CUTOFF, TEST_END)] + [(m, pd.Timestamp(m), pd.Timestamp(m)+pd.offsets.MonthBegin(1)) for m in FOLDS]:
        mask = f.BOUND.isin(['A', 'D']) & f.SCHEDULED_TIME.ge(start-pd.Timedelta(hours=3)) & f.SCHEDULED_TIME.lt(end+pd.Timedelta(hours=4))
        for kind in ['flight', 'airport', 'global']:
            fallback.append(dict(fold=fold, level=kind, n=int((level[mask] == kind).sum()), total=int(mask.sum()), pct=100*np.mean(level[mask] == kind)))
    for typ in ['DA APT', 'DA FS']:
        d = valid[valid.type == typ].sort_values('time').reset_index(drop=True)
        tr = d.time < CUTOFF
        te = d.time.between(CUTOFF, TEST_END, inclusive='left')
        train_counts[typ] = int(tr.sum())
        threshold = d.loc[tr, 'y'].quantile(.9)
        target = d.loc[te, ['time', 'type', 'y']].copy()
        target['fold'] = target.time.dt.strftime('%Y-%m')
        target['peak_threshold'] = threshold

        def record(name, values):
            out = target.copy()
            out['model'] = name
            out['prediction'] = np.maximum(0, values)
            preds.append(out)

        for name in MAIN_MODELS + DIAGNOSTICS:
            x = features(d.time, series[name if name in DIAGNOSTICS else 'PAX'])
            if name == 'CAL':
                x = x[[c for c in x if not c.startswith(('A_', 'D_'))]]
            elif name == 'SCHED':
                x = x[[c for c in x if '_pax_' not in c]]
            reg = estimator(name)
            reg.fit(x.loc[tr], d.loc[tr, 'y'])
            record(name, reg.predict(x.loc[te]))
            fit_audit.append(dict(type=typ, model=name, cutoff=str(CUTOFF), latest_training_time=str(d.loc[tr, 'time'].max()), train_n=int(tr.sum()), feature_count=len(x.columns), columns=list(x.columns)))
            if name in MAIN_MODELS:
                joblib.dump({'estimator': reg, 'columns': list(x.columns), 'cutoff': str(CUTOFF), 'model': name, 'type': typ}, model_dir/f'{typ.replace(" ", "_")}_{MAIN_MODELS.index(name)}.joblib')
            if name == 'PAX':
                train_p = np.maximum(0, reg.predict(x.loc[tr]))
                edges[typ] = np.unique(np.quantile(train_p, np.linspace(0, 1, 6)))
                edges[typ][0], edges[typ][-1] = -np.inf, np.inf
                fit_audit[-1]['calibration_inner_edges'] = edges[typ][1:-1].tolist()
                qr = HistGradientBoostingRegressor(**{**reg.get_params(), 'loss': 'quantile', 'quantile': .9})
                qr.fit(x.loc[tr], d.loc[tr, 'y'])
                qout = target.copy()
                qout['q90'] = np.maximum(0, qr.predict(x.loc[te]))
                qpreds.append(qout)
                fit_audit.append(dict(type=typ, model='Q90', cutoff=str(CUTOFF), latest_training_time=str(d.loc[tr, 'time'].max()), train_n=int(tr.sum()), feature_count=len(x.columns), columns=list(x.columns)))
        frozen = d.loc[tr].groupby(['hour', 'dow']).y.mean()
        global_mean = d.loc[tr, 'y'].mean()
        record('MEAN', frozen.reindex(pd.MultiIndex.from_frame(d.loc[te, ['hour', 'dow']])).fillna(global_mean).to_numpy())
        recent = []
        for row in d.loc[te].itertuples():
            day = row.time.normalize()
            pool = d[(d.time < day) & (d.time >= day-pd.Timedelta(days=28)) & (d.hour == row.hour) & (d.dow == row.dow)]
            recent.append(pool.y.mean() if len(pool) else frozen.get((row.hour, row.dow), global_mean))
            audit.append(dict(type=typ, fold=row.time.strftime('%Y-%m'), time=row.time, target_day=day, recent_n=len(pool), fallback=(len(pool) == 0), latest_source_time=pool.time.max() if len(pool) else pd.NaT))
        record('RECENT', recent)
        for fold, z in periods(target):
            samples.append(dict(type=typ, fold=fold, train_n=int(tr.sum()), test_n=len(z), threshold=threshold, test_days=z.time.dt.normalize().nunique()))
        print(f'{typ}: fitted once on {tr.sum()} hours; tested on {te.sum()} hours', flush=True)
    pred, qpred = pd.concat(preds, ignore_index=True), pd.concat(qpreds, ignore_index=True)
    pred.to_csv(OUT/'predictions.csv', index=False)
    qpred.to_csv(OUT/'quantile_predictions.csv', index=False)
    met = summarize(pred, qpred, train_counts, edges)
    gains = paired(pred)
    gains.to_csv(OUT/'paired_gains.csv', index=False)
    # Keep top-level entry points consistent with the canonical result directory.
    pred.to_csv(ROOT/'results'/'predictions.csv', index=False)
    met.to_csv(ROOT/'results'/'model_metrics.csv', index=False)
    gains.to_csv(ROOT/'results'/'paired_bootstrap.csv', index=False)
    directional(pred).to_csv(OUT/'directional_gains.csv', index=False)
    pd.DataFrame(fallback).to_csv(OUT/'passenger_fallback.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT/'recent_audit.csv', index=False)
    pd.DataFrame(samples).to_csv(OUT/'samples.csv', index=False)
    (OUT/'fit_audit.json').write_text(json.dumps(fit_audit, indent=2)+'\n')
    zero_sensitivity(f, s)
    ss = s[s.type == 'SS'].set_index('time').y
    strata = []
    for (typ, fold, name), z in pred.groupby(['type', 'fold', 'model']):
        z = z.copy()
        z['ss'] = ss.reindex(z.time).to_numpy()
        for label, mask in [('SS=0', z.ss == 0), ('SS>0', z.ss > 0), ('SS absent', z.ss.isna())]:
            if mask.any():
                strata.append(dict(type=typ, fold=fold, model=name, stratum=label, **metrics(z.loc[mask, 'y'], z.loc[mask, 'prediction'], z.peak_threshold.iloc[0])))
    pd.DataFrame(strata).to_csv(ROOT/'results'/'service_strata.csv', index=False)
    meta = {'training_cutoff': str(CUTOFF), 'test_end_exclusive': str(TEST_END), 'test_months': FOLDS, 'fit_rule': 'Once per direction before March; no refitting in testing', 'main_models': MAIN_MODELS, 'secondary_models': ['MEAN', 'RECENT'], 'passenger_estimates': 'Positive completed flights strictly before March; frozen throughout testing', 'recent_window_days': 28, 'recent_update': 'Completed days strictly before target day; fixed training-set MEAN fallback', 'schedule_status': 'Retrospective movement selection; cancellation completeness and input delivery times unknown', 'bootstrap': {'replicates': 1000, 'unit': 'day', 'seed': 42, 'directional_seed': 43, 'between_day_dependence': 'not modeled'}, 'ablation': 'same 24 features; no low-cost/domestic terms', 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    (OUT/'protocol.json').write_text(json.dumps(meta, indent=2)+'\n')
    (ROOT/'report.md').write_text('# Chronological airport shuttle benchmark\n\nTraining: all eligible observations before 2025-03-01. Testing: March--May 2025. Models and passenger averages are fixed throughout testing; only RECENT receives completed-day test counts. No missing targets are imputed in the main comparison.\n\n## Full-holdout scores\n\n'+md(met[met.fold == 'ALL'][['type', 'model', 'train_n', 'n', 'MAE', 'RMSE', 'WAPE_pct', 'R2', 'peak_MAE']])+'\n\nRun `python code/analyze.py` for descriptive analysis and all experiments, then `python code/verify_revision.py` and `python code/verify.py`. Publication assets are generated by `make_paper_figures.py`, `make_revision_figures.py`, and `make_revision_tables.py`.\n', encoding='utf-8')
    compress_outputs()
    return met, pred


if __name__ == '__main__':
    flights, shuttle, _ = load()
    run(flights, shuttle)
