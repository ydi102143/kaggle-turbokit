from __future__ import annotations
from pathlib import Path
import io, base64
import pandas as pd, joblib
import matplotlib.pyplot as plt
from ..utils import ROOT

def _img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def generate_shap_report(cfg: dict, run_id: str, max_samples: int = 5000):
    try:
        import shap
    except Exception:
        raise RuntimeError("shap が未インストールです。pip install shap を実行してください。")
    model_path = ROOT / cfg["output"]["dir"] / "models" / run_id / "model.joblib"
    pipe = joblib.load(model_path)
    est = pipe.named_steps.get("est"); pre = pipe.named_steps.get("pre")
    train = pd.read_csv(Path(cfg["data"]["train"]).resolve())
    target = cfg["data"].get("target"); id_col = cfg["data"].get("id_col")
    X = train.drop(columns=[c for c in [target, id_col] if c in train.columns])
    if len(X) > max_samples: X = X.sample(max_samples, random_state=42)
    Xp = pre.fit_transform(X)

    explainer = shap.Explainer(est); values = explainer(Xp)
    html = "<html><head><meta charset='utf-8'><title>SHAP Report</title></head><body><h1>SHAP</h1>"
    fig = plt.figure(); shap.plots.bar(values, show=False, max_display=30); html += f"<img src='data:image/png;base64,{_img(fig)}'/>"
    fig = plt.figure(); shap.plots.beeswarm(values, show=False, max_display=20); html += f"<img src='data:image/png;base64,{_img(fig)}'/>"
    html += "</body></html>"
    out = ROOT / cfg["output"]["dir"] / "reports" / f"shap_{run_id}.html"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8"); return out
