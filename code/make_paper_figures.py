"""Generate the English vector figures used by paper/main.tex from saved results."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from publication_paths import PAPER, RESULTS
FIG=PAPER/'figures'; FIG.mkdir(exist_ok=True)
RES=RESULTS
C=['#216e95','#df8b29','#715da6','#70869b','#263c50']
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42,'ps.fonttype':42})
def save(name):
    plt.tight_layout(pad=.5,rect=(0,0,1,.83) if name=='errors' else (0,0,1,1)); plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight'); plt.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight'); plt.close()

def main():
    monthly=pd.read_csv(RES/'flights_monthly.csv'); shuttle=pd.read_csv(RES/'shuttle_monthly.csv'); s=pd.read_csv(RES/'shuttle_clean.csv'); corr=pd.read_csv(RES/'lag_correlations.csv')
    fig,ax=plt.subplots(1,2,figsize=(7.05,2.35))
    for i,b in enumerate(['A','D']):
        d=monthly[monthly.BOUND==b]; ax[0].plot(pd.to_datetime(d.month),d.pax/1000,label='Arrivals' if b=='A' else 'Departures',color=C[i],lw=1.25)
    ax[0].axvspan(pd.Timestamp('2025-07-01'),pd.Timestamp('2025-07-18'),color=C[1],alpha=.18)
    ax[0].text(.98,.02,'Jul. 2025: 10 days',transform=ax[0].transAxes,ha='right',fontsize=7)
    ax[0].set(title='(a) Air passengers by month',ylabel='Passengers (thousands)'); ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6)); ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %y')); ax[0].legend(loc='lower left',bbox_to_anchor=(0,.13),frameon=False)
    months=sorted(shuttle.month.unique())
    for i,(typ,label) in enumerate([('DA APT','Airport to station'),('DA FS','Station to airport'),('SS','Replacement services')]):
        d=shuttle[shuttle.type==typ].set_index('month').reindex(months); ax[1].plot(range(len(months)),d.coverage_5_23_pct,marker='o',markersize=3,lw=1.15,color=C[i],label=label)
    ax[1].set(title='(b) Shuttle record coverage',ylabel='Observed hours / reference grid (%)',ylim=(45,104)); ax[1].set_xticks(range(len(months)),[x[2:].replace('-','/') for x in months],rotation=55); ax[1].legend(loc='lower left',frameon=False)
    save('coverage')
    fig,ax=plt.subplots(2,2,figsize=(7.05,3.75))
    for i,(typ,label) in enumerate([('DA APT','Airport to station'),('DA FS','Station to airport')]):
        d=s[(s.type==typ)&s.hour.between(5,23)]; g=d.groupby('hour').y
        ax[0,i].plot(g.mean().index,g.mean(),color=C[i],lw=1.4); ax[0,i].fill_between(g.mean().index,g.quantile(.1),g.quantile(.9),color=C[i],alpha=.15)
        ax[0,i].set(title=f'({chr(97+i)}) {label}',xlabel='Shuttle hour',ylabel='Trips / observed hour'); ax[0,i].set_xticks([5,8,11,14,17,20,23]); ax[0,i].set_ylim(0,340)
        r=corr[corr.type==typ]; ax[1,i].plot(r.offset_hours,r.pearson_raw,'o-',color=C[3],markersize=3,lw=1,label='Raw'); ax[1,i].plot(r.offset_hours,r.pearson_residual,'s-',color=C[i],markersize=3,lw=1,label='Hour-weekday adjusted'); ax[1,i].axvline(0,color='gray',ls=':',lw=.8)
        ax[1,i].set(title=f'({chr(99+i)}) Flight-shuttle association',xlabel='Flight hour minus shuttle hour',ylabel='Pearson correlation',ylim=(-.05,.72)); ax[1,i].set_xticks(range(-4,5,2)); ax[1,i].legend(frameon=False,loc='upper left')
    save('profiles_lags')
    met=pd.read_csv(RES/'model_metrics.csv'); names=['Media storica ora × settimana','Calendario','Calendario + voli programmati','Calendario + pax stimati','Retrospettivo pax e orari effettivi']; labels=['Historical mean','Calendar GB','Schedule counts','Estimated passengers','Actual flights (post hoc)']
    fig,ax=plt.subplots(1,2,figsize=(7.05,2.5))
    for i,(typ,label) in enumerate([('DA APT','Airport to station'),('DA FS','Station to airport')]):
        d=met[met.type==typ].pivot(index='fold',columns='model',values='MAE').reindex(columns=names); d.columns=labels; d.plot.bar(ax=ax[i],color=[C[4],C[3],C[0],C[1],C[2]],width=.82,legend=False)
        ax[i].set(title=f'({chr(97+i)}) {label}',xlabel='Held-out month',ylabel='MAE (trips/hour)',ylim=(0,52)); ax[i].set_xticklabels(['Sep 24','Mar 25','Apr 25','May 25'],rotation=0)
    handles,lab=ax[0].get_legend_handles_labels(); fig.legend(handles,lab,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False); save('errors')
    pred=pd.read_csv(RES/'predictions.csv',parse_dates=['time']); fig,ax=plt.subplots(2,1,figsize=(7.05,2.9),sharex=True)
    for i,(typ,label) in enumerate([('DA APT','Airport to station'),('DA FS','Station to airport')]):
        z=pred[(pred.type==typ)&(pred.fold=='2025-05')&pred.time.lt('2025-05-08')]
        obs=z.drop_duplicates('time').set_index('time').y; grid=pd.date_range('2025-05-01','2025-05-07 23:00',freq='h'); obs=obs.reindex(grid); ax[i].plot(obs.index,obs,color=C[4],lw=.95,label='Observed')
        for name,col,lab in [('Calendario',C[0],'Calendar GB'),('Calendario + pax stimati',C[1],'Estimated passengers')]:
            d=z[z.model==name].set_index('time').prediction.reindex(grid); ax[i].plot(d.index,d,color=col,lw=.9,label=lab)
        ax[i].set(title=f'({chr(97+i)}) {label}',ylabel='Trips/hour',ylim=(0,380)); ax[i].legend(loc='upper right',ncol=3,frameon=False)
    ax[1].xaxis.set_major_locator(mdates.DayLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d May')); save('week')
    print('Generated 4 vector PDF figures and PNG previews.')

if __name__=='__main__': main()
