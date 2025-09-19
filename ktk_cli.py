
import typer
import yaml
from pathlib import Path
from src.core.registry import get_trainer, get_inferer
from src.tune import run_tuning
from src.utils import make_run_id

app = typer.Typer(help="Kaggle TurboKit – speedrun strong baselines")

@app.command()
def train(cfg: str = typer.Option("conf/default.yaml", help="Path to config YAML")):
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    run_id = make_run_id(prefix=config.get("name", "exp"))
    typer.echo(f"[TurboKit] Training run: {run_id} (task={config.get('task')})")
    trainer = get_trainer(config.get("task", "tabular"))
    trainer(config, run_id)
    typer.echo("[TurboKit] Done training.")

@app.command()
def infer(cfg: str = typer.Option("conf/default.yaml", help="Path to config YAML"),
          run_id: str = typer.Option(..., help="Run ID of trained model")):
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    inferer = get_inferer(config.get("task", "tabular"))
    out_path = inferer(config, run_id)
    typer.echo(f"[TurboKit] Saved predictions -> {out_path}")

@app.command()
def tune(cfg: str = typer.Option("conf/default.yaml", help="Path to config YAML"),
         n_trials: int = typer.Option(30, help="Optuna trials")):
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    run_id, best_params, best_score = run_tuning(config, n_trials=n_trials)
    typer.echo(f"[TurboKit] Best run: {run_id} score={best_score}\nParams: {best_params}")

@app.command()
def submit(competition: str = typer.Option(..., help="Kaggle competition slug"),
           csv: str = typer.Option(..., help="Prediction CSV path"),
           message: str = typer.Option("TurboKit submit", help="Submission message"),
           force: bool = typer.Option(False, help="Submit even if already submitted (by file hash)"),
           dry_run: bool = typer.Option(False, help="Log only, do not call Kaggle")):
    from src.submit.submit import submit_csv
    info = submit_csv(competition, csv, message=message, force=force, dry_run=dry_run)
    typer.echo(f"[TurboKit] Submit status: {info.get('status')}")

@app.command()
def blend(files: list[str] = typer.Argument(..., help="Prediction CSVs to blend"),
         method: str = typer.Option("mean", help="mean | gmean | rank")):
    from src.blend import run_blend
    out = run_blend(files, method)
    typer.echo(f"[TurboKit] Blended -> {out}")

@app.command("stack-learn")
def stack_learn(files: list[str] = typer.Argument(..., help="Test prediction CSVs"),
                oof: list[str] = typer.Option(None, help="OOF CSVs aligned with files"),
                task: str = typer.Option("regression", help="regression | classification"),
                meta: str = typer.Option("ridge", help="ridge | lgbm | logreg"),
                out_name: str = typer.Option(None, help="Output filename")):
    from src.ensemble.stack_learn import run_stack_learn
    out = run_stack_learn(files, oof_files=oof, task=task, meta=meta, out_name=out_name)
    typer.echo(f"[TurboKit] Learned stack -> {out}")

@app.command("stack-cv")
def stack_cv(run_ids: list[str] = typer.Argument(..., help="Base run_ids to stack"),
             cfg: str = typer.Option("conf/default.yaml", help="Config for output dir/label"),
             task: str = typer.Option("regression", help="regression | classification"),
             meta: str = typer.Option("ridge", help="ridge | lgbm | logreg"),
             out_name: str = typer.Option(None, help="Output filename")):
    import yaml
    from src.ensemble.stack_cv import run_stack_cv
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    out = run_stack_cv(run_ids, cfg=config, task=task, meta=meta, out_name=out_name)
    typer.echo(f"[TurboKit] Stacked (CV meta) -> {out}")

@app.command("check-data")
def check_data(cfg: str = typer.Option("conf/default.yaml", help="Config YAML for dataset"),
               save_json: bool = typer.Option(True, help="Save report JSON under outputs/logs")):
    import json
    from src.quality.leak_checks import run_checks
    from src.utils import ROOT
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    rep = run_checks(config)
    typer.echo(f"[TurboKit] Data check flags: {rep['summary']['flags']} (ok={rep['summary']['ok']})")
    if save_json:
        out_dir = ROOT / config["output"]["dir"] / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "data_check_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[TurboKit] Saved -> {out_dir / 'data_check_report.json'}")

@app.command("track-submissions")
def track_submissions(competition: str = typer.Option(..., help="Kaggle competition slug")):
    from src.submit.submit import list_submissions
    path = list_submissions(competition)
    typer.echo(f"[TurboKit] Saved Kaggle submissions table -> {path}")

if __name__ == "__main__":
    app()
