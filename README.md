
# Kaggle TurboKit

Speedrun strong baselines for Kaggle tasks (Tabular / TimeSeries / Image / NLP).  
Includes: CV+OOF, stacking, blending, Optuna tuning, data leak checks, logs, and Kaggle submit workflow.

## Quickstart
```bash
pip install -r requirements.txt

# Tabular (sample)
python ktk_cli.py train --cfg conf/default.yaml
python ktk_cli.py infer --cfg conf/default.yaml --run-id <run_id>

# TimeSeries (multi-series sample with history)
python ktk_cli.py train --cfg conf/timeseries_multi.yaml
python ktk_cli.py infer --cfg conf/timeseries_multi.yaml --run-id <run_id>

# Image (toy)
python ktk_cli.py train --cfg conf/image.yaml
python ktk_cli.py infer --cfg conf/image.yaml --run-id <run_id>

# NLP (toy)
python ktk_cli.py train --cfg conf/nlp.yaml
python ktk_cli.py infer --cfg conf/nlp.yaml --run-id <run_id>
```

## Training logs
Each run logs metrics to `outputs/logs/<run_id>/metrics.csv` and events to `events.jsonl`.

## Submit & Track
```bash
python ktk_cli.py submit --competition <slug> --csv outputs/preds/<run_id>.csv --message "v1"
python ktk_cli.py track-submissions --competition <slug>
```
