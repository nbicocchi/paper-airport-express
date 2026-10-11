# Airport Express — demand forecasting

Reproducible experiments on hourly demand for Bologna's Marconi Express airport
rail service. The analysis compares historical and calendar baselines with models
using flight schedules and estimated passenger loads.

## Quick start

Requires Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
python code/download.py --check
python code/analyze.py
python code/verify.py
python code/revision.py
python code/verify_revision.py
```

The 32 Excel inputs and saved results are included. The commands verify the inputs,
rerun the experiments and check the results. No manuscript access is required.

## Repository contents

- `dataset/`: airport and shuttle Excel inputs.
- `code/`: analysis, verification, prediction and publication scripts.
- `code/results/`: saved metrics, predictions and audit records.
- `code/examples/`: example inputs and outputs for prediction.

See [code/README.md](code/README.md) for inference examples, model definitions,
evaluation limitations and publication commands.

These are historical experiments, not a production forecasting service or a
prospective validation of flight-schedule availability.
