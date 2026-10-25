"""Generate main, monthly, and secondary manuscript tables from the holdout."""
import pandas as pd
from publication_paths import PAPER, RESULTS

R = RESULTS/'revision'
T = PAPER/'tables'
B = chr(92)
NEWLINE = B*2+'\n'
DIRECTIONS = ['DA APT', 'DA FS']


def interval(row):
    return f'{row.gain_pct:.1f} [{row.low:.1f}, {row.high:.1f}]'


def main():
    T.mkdir(exist_ok=True)
    m = pd.read_csv(R/'model_metrics.csv')
    g = pd.read_csv(R/'paired_gains.csv')
    d = pd.read_csv(R/'directional_gains.csv')
    primary = ['CAL', 'SCHED', 'PAX']
    diagnostics = ['REAL-LOAD', 'REAL-TIME', 'REAL-BOTH']
    labels = {'REAL-LOAD': B+'shortstack[l]{Actual'+B*2+'passengers}', 'REAL-TIME': 'Block times', 'REAL-BOTH': B+'shortstack[l]{Both'+B*2+'substitutions}'}
    for period, filename in [('ALL', 'holdout_rows.tex'), ('2025-05', 'may_rows.tex')]:
        text = ''
        for model in primary+['MEAN', 'RECENT']+diagnostics:
            if model == 'MEAN':
                text += B+'midrule\n'
            if model == diagnostics[0]:
                text += B+'midrule\n'+B+'multicolumn{7}{c}{After-the-event substitutions in PAX}'+NEWLINE
            cells = [labels.get(model, B+'textsc{'+model+'}')]
            for typ in DIRECTIONS:
                panel = m[(m.fold == period) & (m.type == typ)].set_index('model')
                best = panel.loc[primary, ['MAE', 'WAPE_pct', 'peak_MAE']].min()
                for metric in ['MAE', 'WAPE_pct', 'peak_MAE']:
                    value = panel.loc[model, metric]
                    cell = f'{value:.2f}'
                    if model in primary and abs(value-best[metric]) < 1e-8:
                        cell = B+'textbf{'+cell+'}'
                    cells.append(cell)
            text += ' & '.join(cells)+NEWLINE
        (T/filename).write_text(text+B+'bottomrule\n')

    text = ''
    for fold in ['ALL', '2025-03', '2025-04', '2025-05']:
        for typ, label in [('DA APT', B+'AS'), ('DA FS', B+'SA')]:
            cells = [('Full test' if fold == 'ALL' else fold) if typ == 'DA APT' else '', label]
            for ref in ['CAL', 'SCHED']:
                row = g[(g.fold == fold) & (g.type == typ) & (g.model == 'PAX') & (g.reference == ref)].iloc[0]
                cells.append(interval(row))
            text += ' & '.join(cells)+NEWLINE
        cells = ['', B+'DeltaGain']
        for ref in ['CAL', 'SCHED']:
            row = d[(d.fold == fold) & (d.reference == ref)].iloc[0]
            cells.append(f'{row.difference_pp:.2f} [{row.low:.2f}, {row.high:.2f}]')
        text += ' & '.join(cells)+NEWLINE+B+'addlinespace\n'
    (T/'gain_rows.tex').write_text(text+B+'bottomrule\n')

    text = ''
    for model in ['MEAN', 'RECENT', 'PAX']:
        cells = [B+'textsc{'+model+'}']
        for typ in DIRECTIONS:
            row = m[(m.fold == 'ALL') & (m.type == typ) & (m.model == model)].iloc[0]
            cells += [f'{row.MAE:.2f}', f'{row.WAPE_pct:.2f}']
        text += ' & '.join(cells)+NEWLINE
    (T/'robustness_rows.tex').write_text(text+B+'bottomrule\n')

    samples = pd.read_csv(R/'samples.csv')
    text = ''
    for fold in ['2025-03', '2025-04', '2025-05', 'ALL']:
        cells = ['Full test' if fold == 'ALL' else fold]
        for typ in DIRECTIONS:
            row = samples[(samples.fold == fold) & (samples.type == typ)].iloc[0]
            cells += [f'{row.train_n:,}', f'{row.test_n:,}']
        text += ' & '.join(cells)+NEWLINE
    (T/'sample_rows.tex').write_text(text+B+'bottomrule\n')
    print('Generated holdout, gain, robustness, and sample table rows.')


if __name__ == '__main__':
    main()
