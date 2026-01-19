#!/usr/bin/env python3
import argparse
import subprocess
import os
from pathlib import Path

EVALUATOR_DIR = Path(__file__).parent.resolve()

def evaluate_predictions(preds_path: str, run_id: str | None = None, 
                         instance_ids: str | None = None):
    preds_path = Path(preds_path).resolve()
    
    if not run_id:
        run_id = preds_path.parent.name
    
    output_dir = EVALUATOR_DIR / "eval_results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"evaluating: {preds_path}")
    print(f"run id: {run_id}")
    if instance_ids:
        print(f"instances: {instance_ids}")
    
    cmd = [
        "sb-cli", "submit", "swe-bench_lite", "test",
        "--predictions_path", str(preds_path),
        "--run_id", run_id,
        "-o", str(output_dir),
    ]
    
    if instance_ids:
        cmd.extend(["--instance_ids", instance_ids])
    
    subprocess.run(cmd, check=True)
    print(f"\nresults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_path", help="Path to preds.json")
    parser.add_argument("--run-id", help="Run ID for evaluation")
    parser.add_argument("--ids", help="Comma-separated instance IDs to evaluate (e.g. 'astropy__astropy-12907,django__django-123')")
    args = parser.parse_args()
    
    evaluate_predictions(args.preds_path, args.run_id, args.ids)
