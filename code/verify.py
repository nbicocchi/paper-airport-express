"""Check source reconciliation, temporal evaluation, and saved-model inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from predict import predict
from data_sources import DATA_ROOT, verify_sources
from revision import CUTOFF
from revision_io import read_result
ROOT=Path(__file__).resolve().parent

def verify():
    verify_sources()
    s=pd.read_csv(ROOT/'results'/'shuttle_clean.csv',parse_dates=['time'])
    monthly=pd.read_csv(ROOT/'results'/'shuttle_monthly.csv')
    assert np.isclose(s.y.sum(),monthly.total.sum())
    assert not s.duplicated(['type','time']).any()
    pred=read_result('predictions.csv',parse_dates=['time'])
    met=pd.read_csv(ROOT/'results'/'revision'/'model_metrics.csv')
    for (typ,fold,name),z in pred.groupby(['type','fold','model']):
        m=met[(met.type==typ)&(met.fold==fold)&(met.model==name)].iloc[0]
        assert z.time.dt.strftime('%Y-%m').eq(fold).all()
        assert len(z)==m.n
        assert np.isclose(mean_absolute_error(z.y,z.prediction),m.MAE)
        earlier=s[(s.type==typ)&(s.time<CUTOFF)&s.time.dt.hour.between(5,23)]
        assert len(earlier)==m.train_n
    # Rebuild the entire holdout through the standalone schedule-only interface.
    flights=pd.read_excel(DATA_ROOT/'Aeroporto'/'2025 BLQ.xlsx')
    flights['SCHEDULED_TIME']=pd.to_datetime(flights.SCHEDULED_TIME)
    schedule=flights[flights.BOUND.isin(['A','D'])&flights.SCHEDULED_TIME.between('2025-02-28','2025-06-02')][['SCHEDULED_TIME','FLIGHT_NUMBER','BOUND','ORIDES_CODE']]
    ref=pred[pred.model=='PAX'].sort_values(['time','type']).reset_index(drop=True)
    targets=ref[['time','type']]
    got=predict(schedule,targets)
    assert np.allclose(got.prediction,ref.prediction,rtol=1e-10,atol=1e-8)
    examples=ROOT/'examples'; examples.mkdir(exist_ok=True)
    schedule.to_csv(examples/'schedule_holdout.csv',index=False); targets.to_csv(examples/'targets_holdout.csv',index=False); got.to_csv(examples/'predicted_holdout.csv',index=False)
    summary={'source_shuttle_total_reconciled':float(s.y.sum()),'evaluated_prediction_rows':len(pred),'inference_predictions_checked':len(got),'max_inference_difference':float(np.max(np.abs(got.prediction-ref.prediction))),'chronology_and_metrics':'passed'}
    (ROOT/'results'/'verification.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': verify()
