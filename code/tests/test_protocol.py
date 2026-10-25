"""Small regression tests for feature windows and the passenger cutoff."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import estimate_pax, features
from revision import CUTOFF, periods


class ProtocolTests(unittest.TestCase):
    def test_feature_counts_and_cross_day_offsets(self):
        grid = pd.date_range('2025-03-01', periods=72, freq='h')
        series = {key: pd.Series(np.arange(len(grid)), index=grid) for key in ['A_count', 'D_count', 'A_pax', 'D_pax']}
        target = pd.DatetimeIndex(['2025-03-01 23:00'])
        x = features(target, series)
        self.assertEqual(len(x.columns), 24)
        self.assertEqual(len([c for c in x if '_pax_' not in c]), 15)
        self.assertEqual(len(features(target, {}).columns), 6)
        self.assertEqual(x.loc[0, 'A_count_-3'], 20)
        self.assertEqual(x.loc[0, 'D_count_+4'], 27)
        self.assertNotIn('A_count_+1', x)
        self.assertNotIn('D_count_-1', x)

    def test_passenger_history_is_strictly_before_cutoff(self):
        # Five positive observations make a flight-specific average reliable.
        f = pd.DataFrame({'FLIGHT_NUMBER': ['F1']*5+['F2', 'F3', 'F1'], 'BOUND': ['A']*8, 'ORIDES_CODE': ['AAA']*6+['BBB', 'AAA'], 'pax': [100, 110, 120, 130, 140, 0, 9999, 9999], 'BLOCK_TIME': [CUTOFF-pd.Timedelta(days=i+1) for i in range(6)]+[CUTOFF, CUTOFF+pd.Timedelta(days=1)]})
        expected = estimate_pax(f, CUTOFF)
        self.assertEqual(expected.iloc[0], 120)
        self.assertEqual(expected.iloc[5], 120)  # Route fallback; zero excluded.
        self.assertEqual(expected.iloc[6], 120)  # Global fallback; cutoff excluded.
        changed = f.copy()
        changed.loc[changed.BLOCK_TIME >= CUTOFF, 'pax'] = 1_000_000
        pd.testing.assert_series_equal(expected, estimate_pax(changed, CUTOFF))

    def test_full_period_reuses_monthly_rows(self):
        frame = pd.DataFrame({'fold': ['2025-03', '2025-04', '2025-05'], 'value': [1, 2, 3]})
        panels = dict(periods(frame))
        self.assertEqual(len(panels['ALL']), 3)
        self.assertEqual(sum(len(panels[m]) for m in ['2025-03', '2025-04', '2025-05']), 3)


if __name__ == '__main__':
    unittest.main()
