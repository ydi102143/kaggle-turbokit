
import subprocess, sys, json, os, time, pathlib

def test_tabular_train_infer_smoke():
    root = pathlib.Path(__file__).resolve().parents[1]
    cli = root / "ktk_cli.py"
    cfg = root / "conf" / "default.yaml"
    # train
    p = subprocess.run([sys.executable, str(cli), "train", "--cfg", str(cfg)], check=True, capture_output=True, text=True)
    assert "[TurboKit] Training run:" in p.stdout
    # parse the last run_id from stdout
    run_id = p.stdout.split("Training run:")[1].split("(")[0].strip()
    # infer
    p2 = subprocess.run([sys.executable, str(cli), "infer", "--cfg", str(cfg), "--run-id", run_id], check=True, capture_output=True, text=True)
    assert "Saved predictions" in p2.stdout
