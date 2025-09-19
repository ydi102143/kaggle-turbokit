
from __future__ import annotations
from pathlib import Path
import subprocess, time, json, hashlib
from typing import Optional, Dict, Any
from src.utils import ROOT

LOG_DIR = ROOT / "outputs" / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "submissions.jsonl"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _append_log(event: Dict[str, Any]) -> None:
    event["ts"] = int(time.time())
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def already_submitted(file_hash: str, competition: Optional[str] = None) -> bool:
    if not LOG_FILE.exists(): return False
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("file_hash") == file_hash and (competition is None or rec.get("competition") == competition):
                    return True
            except Exception: continue
    return False

def submit_csv(competition: str, csv_path: str, message: str = "TurboKit submit",
               force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    path = Path(csv_path); 
    if not path.exists(): raise FileNotFoundError(f"CSV not found: {csv_path}")
    file_hash = _sha256(path)
    if not force and already_submitted(file_hash, competition):
        info = {"status": "skipped-duplicate", "competition": competition, "csv": str(path), "file_hash": file_hash}
        _append_log(info); return info
    cmd = ["kaggle", "competitions", "submit", "-c", competition, "-f", str(path), "-m", message]
    if dry_run:
        info = {"status": "dry-run", "competition": competition, "csv": str(path), "file_hash": file_hash, "cmd": cmd}
        _append_log(info); return info
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        info = {"status": "submitted", "competition": competition, "csv": str(path),
                "file_hash": file_hash, "message": message, "stdout": out}
        _append_log(info); return info
    except subprocess.CalledProcessError as e:
        err = (e.stdout or "") + "\n" + (e.stderr or "")
        info = {"status": "error", "competition": competition, "csv": str(path),
                "file_hash": file_hash, "message": message, "error": err.strip()}
        _append_log(info); raise

def list_submissions(competition: str) -> str:
    out_csv = LOG_DIR / f"kaggle_submissions_{competition}.csv"
    cmd = ["kaggle", "competitions", "submissions", "-c", competition, "-v"]
    with open(out_csv, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, check=True)
    return str(out_csv)
