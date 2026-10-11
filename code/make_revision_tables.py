"""Generate manuscript tables from saved revision results."""
from pathlib import Path
import pandas as pd
from publication_paths import PAPER, RESULTS
R=RESULTS/'revision'
T=PAPER/'tables';T.mkdir(exist_ok=True)
B=chr(92);newline=B*2+'\n'
def main():
 m=pd.read_csv(R/'model_metrics.csv');g=pd.read_csv(R/'paired_gains.csv');d=pd.read_csv(R/'directional_gains.csv')
 models=['MEAN','CAL','CAL-REG','RECENT','SCHED','PAX','RIDGE','REAL-LOAD','REAL-TIME','REAL-BOTH']
 directions=['DA APT','DA FS'];s=''
 for model in models:
  if model=='REAL-LOAD':s+=B+'midrule\n'+B+'multicolumn{7}{c}{Post-hoc, matched-feature ablations (unavailable at issue time)}'+newline
  cells=[B+'textsc{'+model+'}']
  for typ in directions:
   panel=m[(m.fold=='2025-05')&(m.type==typ)].set_index('model');best=panel.loc[models[:7],['MAE','WAPE_pct','peak_MAE']].min()
   for metric in ['MAE','WAPE_pct','peak_MAE']:
    val=panel.loc[model,metric];text=f'{val:.2f}'
    if model in models[:7] and abs(val-best[metric])<1e-8:text=B+'textbf{'+text+'}'
    cells.append(text)
  s+=' & '.join(cells)+newline
 (T/'may_rows.tex').write_text(s+B+'bottomrule\n')
 def interval(a):return f"{a.gain_pct:.1f} [{a.low:.1f}, {a.high:.1f}]"
 s=''
 for fold in ['2024-09','2025-03','2025-04','2025-05']:
  for typ,label in [('DA APT',B+'AS'),('DA FS',B+'SA')]:
   cells=[fold if typ=='DA APT' else '',label]
   for ref in ['CAL','MEAN','SCHED']:
    a=g[(g.fold==fold)&(g.type==typ)&(g.model=='PAX')&(g.reference==ref)].iloc[0];cells.append(interval(a))
   s+=' & '.join(cells)+newline
  cells=['',B+'DeltaGain']
  for ref in ['CAL','MEAN','SCHED']:
   a=d[(d.fold==fold)&(d.reference==ref)].iloc[0];cells.append(f'{0.0 if abs(a.difference_pp)<.05 else a.difference_pp:.1f} [{a.low:.1f}, {a.high:.1f}]')
  s+=' & '.join(cells)+newline+B+'addlinespace\n'
 (T/'gain_rows.tex').write_text(s+B+'bottomrule\n')
 print('Generated data-backed May and paired-gain table rows.')
if __name__=='__main__':main()
