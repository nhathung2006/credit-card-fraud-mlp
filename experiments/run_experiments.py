"""Chạy tuần tự các cấu hình thực nghiệm TOML và ghi kết quả vào artifacts/results_summary.csv."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = PROJECT_ROOT / "artifacts" / "results_summary.csv"


def run_one(config_path: Path, extra_args: list[str]):
    name = config_path.stem
    val_preds = PROJECT_ROOT / "artifacts" / f"val_{name}.csv"
    cmd = [
        sys.executable,
        "-m",
        "fraud_mlp.train",
        "--config",
        str(config_path),
        "--save-val-predictions",
        str(val_preds),
        *extra_args,
    ]
    print(f"\n{'=' * 72}\nRunning {name} ({config_path})\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise SystemExit(f"Experiment {name} failed with exit code {result.returncode}")

    import tomllib

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    ckpt_path = (config_path.resolve().parent / cfg["paths"]["checkpoint"]).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    row = {
        "experiment": name,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "val_predictions": str(val_preds.relative_to(PROJECT_ROOT)),
        "best_epoch": int(ckpt["epoch"]),
        "val_roc_auc": float(ckpt["val_roc_auc"]),
        "val_recall_05": float(ckpt.get("val_recall_05", float("nan"))),
        "elapsed_sec": round(elapsed, 1),
    }
    update_summary(row)
    print(
        f"--> Finished {name} in {elapsed:.1f}s | Epoch {row['best_epoch']}"
        f" | Val ROC-AUC: {row['val_roc_auc']:.6f} | Recall@0.5: {row['val_recall_05']:.6f}",
        flush=True,
    )


def update_summary(row: dict):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df[df["experiment"] != row["experiment"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(SUMMARY_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="Experiment TOML files to run")
    args, extra = parser.parse_known_args()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    probe = SUMMARY_CSV.parent / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    for cfg in args.configs:
        run_one(cfg.resolve(), extra)

    if SUMMARY_CSV.exists():
        print("\n=== SUMMARY TABLE ===", flush=True)
        df = pd.read_csv(SUMMARY_CSV)
        print(df[["experiment", "best_epoch", "val_roc_auc", "val_recall_05", "elapsed_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
