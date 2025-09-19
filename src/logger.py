
from __future__ import annotations
from pathlib import Path
import csv, json, time
from typing import Any

class RunLogger:
    def __init__(self, run_id: str, out_dir: Path):
        self.run_id = run_id
        self.dir = Path(out_dir) / "logs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.dir / "metrics.csv"
        self.jsonl_file = self.dir / "events.jsonl"

    def log(self, step: int, **metrics: Any):
        ts = int(time.time())
        row = {"ts": ts, "step": step, **metrics}
        write_header = not self.csv_file.exists()
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def path(self) -> str:
        return str(self.dir)
