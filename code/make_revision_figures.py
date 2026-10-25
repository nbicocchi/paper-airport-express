"""English vector diagnostics from results/revision/; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_paper_figures import C, save
from publication_paths import RESULTS
from revision_io import read_result
R=RESULTS/'revision'
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})

def main():
    m=pd.read_csv(R/'model_metrics.csv')
    fig,ax=plt.subplots(1,2,figsize=(7.05,2.5))
    for i,(typ,title) in enumerate([('DA APT','Airport to station'),('DA FS','Station to airport')]):
        p=m[(m.type==typ)&(m.fold!='ALL')].pivot(index='fold',columns='model',values='MAE')[['CAL','SCHED','PAX']]
        p.plot.bar(ax=ax[i],color=[C[3],C[0],C[1]],legend=False,width=.82)
        ax[i].set(title=title,ylabel='MAE (trips/hour)',xlabel='Held-out month');ax[i].set_xticklabels(['Mar 25','Apr 25','May 25'],rotation=0)
    h,l=ax[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.87));
    from make_paper_figures import FIG
    fig.savefig(FIG/'revision_errors.pdf',bbox_inches='tight');fig.savefig(FIG/'revision_errors.png',bbox_inches='tight',dpi=200);plt.close(fig)
    cal=pd.read_csv(R/'calibration.csv');p=read_result('predictions.csv')
    fig,ax=plt.subplots(2,2,figsize=(7.05,3.55))
    for i,(typ,title) in enumerate([('DA APT','Airport to station'),('DA FS','Station to airport')]):
        for j,(fold,d) in enumerate(cal[(cal.type==typ)&(cal.fold!='ALL')].groupby('fold')):
            ax[0,i].plot(d.predicted_mean,d.observed_mean,'o-',markersize=3,lw=.9,label=fold,color=C[j])
        ax[0,i].plot([0,310],[0,310],color='gray',ls=':',lw=.9);ax[0,i].set(title=title,xlabel='Mean PAX prediction',ylabel='Mean observed trips',xlim=(0,300),ylim=(0,300));ax[0,i].legend(frameon=False,ncol=2)
        for name,col in [('CAL',C[3]),('PAX',C[1])]:
            z=p[(p.type==typ)&(p.model==name)];e=np.sort((z.prediction-z.y)[z.y>=z.peak_threshold]);ax[1,i].plot(e,np.arange(1,len(e)+1)/len(e),label=name,color=col,lw=1.2)
        ax[1,i].axvline(0,color='gray',ls=':',lw=.8);ax[1,i].set(xlabel='Test-period peak error (prediction − observed)',ylabel='Empirical cumulative probability',ylim=(0,1.02));ax[1,i].legend(frameon=False)
    save('calibration_peaks')
    print('Generated revision_errors and calibration_peaks.')

if __name__=='__main__':main()
