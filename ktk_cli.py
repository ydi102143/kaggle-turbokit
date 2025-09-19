import typer, yaml
from pathlib import Path
from src.core.registry import get_trainer, get_inferer
from src.tune import run_tuning
from src.utils import make_run_id, ROOT

app = typer.Typer(help="Kaggle TurboKit  speedrun strong baselines")

@app.command()
def train(cfg: str = typer.Option("conf/default.yaml", help="Config YAML")):
    import yaml
    from src.core.registry import get_trainer
    from src.utils import make_run_id
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    run_id = make_run_id(prefix=config.get("name", "exp"))
    typer.echo(f"[TurboKit] Training run: {run_id} (task={config.get('task')})")
    trainer = get_trainer(config.get("task", "tabular"))
    trainer(config, run_id)
    typer.echo("[TurboKit] Done training.")

@app.command()
def infer(cfg: str = typer.Option("conf/default.yaml", help="Config YAML"),
          run_id: str = typer.Option(..., help="Run ID")):
    import yaml
    from src.core.registry import get_inferer
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    inferer = get_inferer(config.get("task", "tabular"))
    out_path = inferer(config, run_id)
    typer.echo(f"[TurboKit] Saved predictions -> {out_path}")

@app.command()
def tune(cfg: str = typer.Option("conf/default.yaml", help="Config YAML"),
         n_trials: int = typer.Option(30, help="Optuna trials")):
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    run_id, best_params, best_score = run_tuning(config, n_trials=n_trials)
    typer.echo(f"[TurboKit] Best run: {run_id} score={best_score}\nParams: {best_params}")

@app.command()
def submit(competition: str = typer.Option(...), csv: str = typer.Option(...),
           message: str = typer.Option("TurboKit submit"),
           force: bool = typer.Option(False), dry_run: bool = typer.Option(False)):
    from src.submit.submit import submit_csv
    info = submit_csv(competition, csv, message=message, force=force, dry_run=dry_run)
    typer.echo(f"[TurboKit] Submit status: {info.get('status')}")

@app.command()
def blend(files: list[str] = typer.Argument(...), method: str = typer.Option("mean")):
    from src.blend import run_blend
    out = run_blend(files, method)
    typer.echo(f"[TurboKit] Blended -> {out}")

@app.command("stack-learn")
def stack_learn(files: list[str] = typer.Argument(...),
                oof: list[str] = typer.Option(None),
                task: str = typer.Option("regression"),
                meta: str = typer.Option("ridge"),
                out_name: str = typer.Option(None)):
    from src.ensemble.stack_learn import run_stack_learn
    out = run_stack_learn(files, oof_files=oof, task=task, meta=meta, out_name=out_name)
    typer.echo(f"[TurboKit] Learned stack -> {out}")

@app.command("stack-cv")
def stack_cv(run_ids: list[str] = typer.Argument(...),
             cfg: str = typer.Option("conf/default.yaml"),
             task: str = typer.Option("regression"),
             meta: str = typer.Option("ridge"),
             out_name: str = typer.Option(None)):
    import yaml
    from src.ensemble.stack_cv import run_stack_cv
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    out = run_stack_cv(run_ids, cfg=config, task=task, meta=meta, out_name=out_name)
    typer.echo(f"[TurboKit] Stacked (CV meta) -> {out}")

@app.command("check-data")
def check_data(cfg: str = typer.Option("conf/default.yaml"), save_json: bool = typer.Option(True)):
    import json
    from src.quality.leak_checks import run_checks
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    rep = run_checks(config)
    typer.echo(f"[TurboKit] Data check flags: {rep['summary']['flags']} (ok={rep['summary']['ok']})")
    if save_json:
        out_dir = ROOT / config["output"]["dir"] / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "data_check_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(f"[TurboKit] Saved -> {out_dir / 'data_check_report.json'}")

@app.command("track-submissions")
def track_submissions(competition: str = typer.Option(...)):
    from src.submit.submit import list_submissions
    path = list_submissions(competition)
    typer.echo(f"[TurboKit] Saved Kaggle submissions table -> {path}")

@app.command("viz-logs")
def viz_logs_cmd(run_id: str = typer.Option("latest")):
    from src.viz.logs import viz_logs
    out = viz_logs(str(ROOT), run_id=run_id)
    typer.echo(f"[TurboKit] Wrote report: {out['report']}")

@app.command("eda-report")
def eda_report(cfg: str = typer.Option("conf/default.yaml"),
               sample: int = typer.Option(None)):
    from src.eda.report import generate_report
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    out_dir = ROOT / config["output"]["dir"] / "reports"
    path = generate_report(config, out_dir=out_dir, sample=sample)
    typer.echo(f"[TurboKit] EDA report -> {path}")

@app.command("threshold")
def threshold_opt(run_id: str = typer.Option(...), metric: str = typer.Option("f1"), step: float = typer.Option(0.001),
                  out_name: str = typer.Option(None), cfg: str = typer.Option("conf/default.yaml")):
    from src.postprocess.threshold import optimize_threshold
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    out = optimize_threshold(config, run_id, metric=metric, step=step, out_name=out_name)
    typer.echo(f"[TurboKit] Thresholded submission -> {out}")

@app.command("calibrate")
def calibrate_probs(run_id: str = typer.Option(...), method: str = typer.Option("platt"),
                    cfg: str = typer.Option("conf/default.yaml")):
    from src.postprocess.calibration import fit_and_apply_calibration
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    out = fit_and_apply_calibration(config, run_id, method=method)
    typer.echo(f"[TurboKit] Calibrated submission -> {out}")

@app.command("adv-val")
def adv_val(cfg: str = typer.Option("conf/default.yaml")):
    from src.quality.adversarial import run_adversarial_validation
    config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
    rep = run_adversarial_validation(config)
    typer.echo(f"[TurboKit] Adversarial AUC={rep['auc']:.4f}")

@app.command("blend-robust")
def blend_robust(files: list[str] = typer.Argument(...), method: str = typer.Option("mean"),
                 winsor: float = typer.Option(0.0)):
    from src.blend_robust import run_blend_robust
    out = run_blend_robust(files, method=method, winsor=winsor)
    typer.echo(f"[TurboKit] Robust blend -> {out}")

if __name__ == "__main__":
    app()
