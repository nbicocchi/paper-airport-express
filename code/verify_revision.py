"""Verify the single training cutoff, shared test hours, and all saved scores."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from revision import CUTOFF, TEST_END, FOLDS, MODELS, periods, directional
from revision_io import RESULTS as R, read_result


def main():
    p = read_result('predictions.csv', parse_dates=['time'])
    m = pd.read_csv(R/'model_metrics.csv')
    s = pd.read_csv(R.parent/'shuttle_clean.csv', parse_dates=['time'])
    assert not p.duplicated(['time', 'type', 'model']).any()
    assert set(p.model) == set(MODELS)
    assert set(p.fold) == set(FOLDS)
    assert p.time.between(CUTOFF, TEST_END, inclusive='left').all()
    assert p.time.dt.strftime('%Y-%m').eq(p.fold).all()
    for fold, panel in periods(p):
        for (typ, model), z in panel.groupby(['type', 'model']):
            row = m[(m.type == typ) & (m.fold == fold) & (m.model == model)].iloc[0]
            training = s[(s.type == typ) & (s.time < CUTOFF) & s.time.dt.hour.between(5, 23)]
            assert len(training) == row.train_n
            assert np.isclose(training.y.quantile(.9), z.peak_threshold.iloc[0])
            assert len(z) == row.n
            assert np.isclose(mean_absolute_error(z.y, z.prediction), row.MAE)
            assert np.isclose(100*abs(z.y-z.prediction).sum()/z.y.sum(), row.WAPE_pct)
            assert z.prediction.ge(0).all()
            assert set(z.time) == set(panel[(panel.type == typ) & (panel.model == 'PAX')].time)
    fits = json.loads((R/'fit_audit.json').read_text())
    assert len(fits) == 14
    assert len({(r['type'], r['model']) for r in fits}) == len(fits)
    for row in fits:
        assert pd.Timestamp(row['cutoff']) == CUTOFF
        assert pd.Timestamp(row['latest_training_time']) < CUTOFF
        assert row['feature_count'] == {'CAL': 6, 'SCHED': 15}.get(row['model'], 24)
    a = read_result('recent_audit.csv', parse_dates=['time', 'target_day', 'latest_source_time'])
    assert a.target_day.eq(a.time.dt.normalize()).all()
    used = a.latest_source_time.notna()
    assert (a.loc[used, 'latest_source_time'] < a.loc[used, 'target_day']).all()
    assert (a.loc[used, 'latest_source_time'] >= a.loc[used, 'target_day']-pd.Timedelta(days=28)).all()
    assert a.fallback.eq(a.recent_n.eq(0)).all()
    # Independently reconstruct both historical references from the raw targets.
    for typ in ['DA APT', 'DA FS']:
        d = s[(s.type == typ) & s.time.dt.hour.between(5, 23)]
        tr = d[d.time < CUTOFF]
        frozen = tr.groupby(['hour', 'dow']).y.mean()
        expected = []
        for row in a[a.type == typ].itertuples():
            pool = d[(d.time < row.target_day) & (d.time >= row.target_day-pd.Timedelta(days=28)) & (d.hour == row.time.hour) & (d.dow == row.time.dayofweek)]
            assert len(pool) == row.recent_n
            expected.append(pool.y.mean() if len(pool) else frozen.get((row.time.hour, row.time.dayofweek), tr.y.mean()))
        actual = p[(p.type == typ) & (p.model == 'RECENT')].sort_values('time')
        assert np.allclose(actual.prediction, expected)
    g = pd.read_csv(R/'paired_gains.csv')
    assert (g.low <= g.high).all()
    for row in g.itertuples():
        panel = m[(m.type == row.type) & (m.fold == row.fold)].set_index('model')
        assert np.isclose(row.gain_pct, 100*(1-panel.loc[row.model, 'MAE']/panel.loc[row.reference, 'MAE']))
    saved = pd.read_csv(R/'directional_gains.csv')
    pd.testing.assert_frame_equal(saved, directional(p), check_dtype=False, atol=1e-10, rtol=1e-10)
    calibration = pd.read_csv(R/'calibration.csv')
    for fit in [r for r in fits if r['model'] == 'PAX']:
        edges = [-np.inf, *fit['calibration_inner_edges'], np.inf]
        for fold, panel in periods(p[(p.type == fit['type']) & (p.model == 'PAX')]):
            bins = pd.cut(panel.prediction, edges, labels=False, include_lowest=True)
            for group, z in panel.groupby(bins):
                row = calibration[(calibration.type == fit['type']) & (calibration.fold == fold) & (calibration.bin == group)].iloc[0]
                assert len(z) == row.n
                assert np.isclose(z.prediction.mean(), row.predicted_mean)
                assert np.isclose(z.y.mean(), row.observed_mean)
    peak = pd.read_csv(R/'peak_diagnostics.csv')
    for fold, panel in periods(p):
        for (typ, model), z in panel.groupby(['type', 'model']):
            row = peak[(peak.type == typ) & (peak.fold == fold) & (peak.model == model)].iloc[0]
            e = (z.prediction-z.y)[z.y >= z.peak_threshold]
            assert len(e) == row.peak_n
            assert np.isclose(e.mean(), row.peak_bias)
            assert np.isclose(100*(e < 0).mean(), row.under_pct)
    q = read_result('quantile_predictions.csv', parse_dates=['time'])
    qm = pd.read_csv(R/'quantile_metrics.csv')
    for fold, panel in periods(q):
        for typ, z in panel.groupby('type'):
            row = qm[(qm.type == typ) & (qm.fold == fold)].iloc[0]
            assert len(z) == row.n
            assert np.isclose(100*(z.y <= z.q90).mean(), row.q90_coverage)
            assert np.isclose(mean_pinball_loss(z.y, z.q90, alpha=.9), row.q90_pinball)
    sensitivity = pd.read_csv(R.parent/'zero_sensitivity.csv')
    assert set(sensitivity.fold) == {'ALL', *FOLDS}
    assert sensitivity.train_n.eq(2907).all()
    summary = dict(prediction_rows=len(p), models=int(p.model.nunique()), test_months=FOLDS, fits=len(fits), paired_gain_rows=len(g), checks='single training cutoff; 6/15/24 features; aggregate and monthly scores; shared hours; independent RECENT reconstruction; directional intervals; peak and quantile scores')
    (R/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
