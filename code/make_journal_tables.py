"""Build the journal supplement from saved, verified revision results."""
from pathlib import Path
import pandas as pd
from publication_paths import PAPER, RESULTS
from revision_io import read_result

R = RESULTS / 'revision'
P = PAPER
B = chr(92)

def latex_table(frame, columns, caption, labels=None, formats=None):
    labels = labels or {}
    formats = formats or {}
    lines = [B+'begin{table}[H]', B+'centering'+B+'footnotesize',
             B+'caption{'+caption+'}', B+'begin{tabular}{'+('l'*len(columns))+'}',
             B+'toprule', ' & '.join(labels.get(c,c) for c in columns)+B*2, B+'midrule']
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for c, v in zip(columns, row):
            if isinstance(v, float):
                values.append(format(v, formats.get(c,'.2f')))
            else:
                values.append(str(v).replace('DA APT', B+'AS').replace('DA FS', B+'SA').replace('_', B+'_'))
        lines.append(' & '.join(values)+B*2)
    lines += [B+'bottomrule', B+'end{tabular}', B+'end{table}']
    return '\n'.join(lines)+'\n'

def main():
    metrics=pd.read_csv(R/'model_metrics.csv')
    gains=pd.read_csv(R/'paired_gains.csv')
    parts=[r'''\documentclass[preprint,11pt,times]{elsarticle}
\usepackage[T1]{fontenc}
\usepackage{courier}
\usepackage{amsmath,booktabs,graphicx,microtype,float}
\setlength{\tabcolsep}{4pt}
\usepackage[hidelinks]{hyperref}
\newcommand{\AS}{\ensuremath{A\!\rightarrow\!S}}
\newcommand{\SA}{\ensuremath{S\!\rightarrow\!A}}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{frontmatter}
\title{Supplementary material: When do flight schedules improve airport rail demand forecasts?}
\author{Author information pending completion}
\begin{abstract}
Complete chronological comparisons and audits for the recorded ordinary-service targets.
Tables are generated from the saved revision CSVs; no model is fitted by this script.
Intervals are paired whole-day percentile bootstrap summaries, not multiplicity-adjusted tests.
\end{abstract}
\end{frontmatter}
\section{Complete accuracy panels}
Counts are recorded trips per hour. Main comparisons use identical target hours within each month and direction.
REAL models are matched-feature post-hoc substitutions. ACTUAL-EXTRA is an archived exploratory model and is excluded from these panels.
''']
    models=['MEAN','CAL','CAL-REG','RECENT','SCHED','PAX','RIDGE','REAL-LOAD','REAL-TIME','REAL-BOTH']
    for fold in ['2024-09','2025-03','2025-04','2025-05']:
        f=metrics[(metrics.fold==fold)&metrics.model.isin(models)].copy()
        f['model']=pd.Categorical(f.model,models,ordered=True)
        f=f.sort_values(['type','model'])
        parts.append(latex_table(f,['type','model','n','MAE','RMSE','WAPE_pct','R2'],
            fold+' complete point-model errors. WAPE is a percentage; all other error magnitudes are trips/hour.',
            {'type':'Direction','model':'Model','n':'Hours','WAPE_pct':'WAPE','R2':r'$R^2$'}))
        parts.append(B+'clearpage\n')
    parts.append(r'\section{Incremental gains and directional differences}'+'\n')
    labels={'type':'Dir.','fold':'Month','reference':'Reference','gain_pct':'Gain','low':r'95\% lower','high':r'95\% upper','model':'Model','difference_pp':r'$\Delta$ (pp)','common_hours':'Hours'}
    f=gains[(gains.model=='PAX')&gains.reference.isin(['CAL','MEAN','SCHED'])]
    parts.append(latex_table(f,['fold','type','reference','gain_pct','low','high'],
        'PAX relative MAE gains against the original references, all months (percent). Positive favors PAX.',labels))
    parts.append(B+'clearpage\n')
    f=gains[(gains.model=='PAX')&gains.reference.isin(['CAL-REG','RECENT'])]
    parts.append(latex_table(f,['fold','type','reference','gain_pct','low','high'],
        'PAX relative MAE gains against stronger calendar and recent-history references (percent). RECENT has daily target-history updates.',labels))
    f=pd.read_csv(R/'directional_gains.csv')
    parts.append(latex_table(f,['fold','reference','difference_pp','low','high','common_hours'],
        'PAX gain toward the station minus its gain toward the airport, on common hours with the same day draws. Units are percentage points.',labels))
    parts.append(B+'clearpage\n')
    f=gains[gains.reference=='PAX']
    parts.append(latex_table(f,['fold','type','model','gain_pct','low','high'],
        'Matched-feature realized-information gains relative to PAX (percent). Negative values mean deterioration; all models are post hoc.',labels))
    parts.append(B+'clearpage\n')
    parts.append(r'\section{Load fallback and peak diagnostics}'+'\n')
    f=pd.read_csv(R/'passenger_fallback.csv')
    parts.append(latex_table(f,['fold','bound','level','n','total','pct'],
        'Historical load-estimation fallback on recorded scheduled test-month movements, with three preceding and four following boundary hours. Counts are flights; A/D are arrivals/departures.',
        {'fold':'Month','bound':'Bound','level':'Pooling','n':'Flights','total':'Total','pct':'Percent'}))
    f=pd.read_csv(R/'peak_diagnostics.csv');f=f[f.model=='PAX']
    f['direction_month']=f.type.map({'DA APT':'A-S','DA FS':'S-A'})+' '+f.fold
    parts.append(latex_table(f,['direction_month','peak_n','peak_bias','peak_MAE','under_pct','error_p10','error_p50','error_p90'],
        'PAX errors in realized peaks, defined by the training 90th-percentile target. Signed error is prediction minus observation; negative means underprediction.',
        {'direction_month':'Dir./month','peak_n':'N','peak_bias':'Bias','peak_MAE':'MAE','under_pct':r'Under (\%)','error_p10':'P10','error_p50':'P50','error_p90':'P90'}))
    parts.append(B+'clearpage\n')
    parts.append(r'\section{Calibration, quantiles and adaptive reference coverage}'+'\n')
    f=pd.read_csv(R/'calibration.csv');f['bin_id']=f.groupby(['type','fold']).cumcount()+1
    parts.append(latex_table(f,['fold','type','bin_id','n','predicted_mean','observed_mean','bias'],
        'PAX calibration bins. Boundaries are quintiles of training predictions, applied unchanged to the test month; ordered bin IDs run from low to high. Bias is predicted minus observed mean.',
        {'fold':'Month','type':'Dir.','bin_id':'Bin','n':'N','predicted_mean':'Predicted','observed_mean':'Observed','bias':'Bias'}))
    parts.append(B+'clearpage\n')
    f=pd.read_csv(R/'quantile_metrics.csv')
    parts.append(latex_table(f,['fold','type','n','q90_coverage','q90_pinball','point_pinball','peak_coverage'],
        'Upper-quantile coverage (percent) and pinball loss at 0.90. Realized-peak coverage is descriptive and is not the nominal unconditional calibration target.',
        {'fold':'Month','type':'Dir.','n':'N','q90_coverage':'Coverage','q90_pinball':'Q90 loss','point_pinball':'Point loss','peak_coverage':'Peak cover.'}))
    audit=read_result('recent_audit.csv',parse_dates=['time','issue_time','latest_source_time'])
    recent=audit.groupby(['fold','type']).agg(n=('time','size'),fallback_pct=('fallback',lambda x:100*x.mean()),mean_recent_n=('recent_n','mean')).reset_index()
    cv=pd.read_csv(R/'calendar_cv.csv').sort_values('inner_MAE').groupby(['fold','type']).first().reset_index()
    f=recent.merge(cv[['fold','type','alpha']],on=['fold','type'])
    parts.append(latex_table(f,['fold','type','n','fallback_pct','mean_recent_n','alpha'],
        'RECENT fallback rate and mean same-hour/weekday observations in its preceding 28-day window; final column gives CAL-REG penalty chosen strictly within pre-month data.',
        {'fold':'Month','type':'Dir.','n':'N','fallback_pct':r'Fallback (\%)','mean_recent_n':'Recent N','alpha':r'$\alpha$'}))
    parts.append(r'''\section{Machine-readable reproducibility records}
The repository includes the per-hour point and quantile predictions, recent-history source-time audit, inner calendar validation scores and protocol metadata.
Point-model metrics, peak counts, gains and quantile coverage are recomputed by \texttt{verify\_revision.py}; the original predictions remain separately archived.
No test target is zero-filled in the main benchmark. The separate zero-fill scenario is a sensitivity assumption and is not certified ground truth.
\end{document}
''')
    (P/'supplement.tex').write_text('\n'.join(parts))
    print('Generated journal supplement: 13 data-backed tables.')

if __name__=='__main__':
    main()
