#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import yaml

from utils import (
    SWEAGENT_ROOT, RESULTS_DIR, CODER_RESULTS_DIR, 
    TRAJECTORIES_DIR, CODER_TRAJECTORIES_DIR,
    DEFAULT_CONFIGS, resolve_model, resolve_path
)

sys.path.insert(0, str(SWEAGENT_ROOT))
from sweagent.agent.agents import DefaultAgentConfig
from sweagent.run.run_batch import RunBatch, RunBatchConfig
from sweagent.run.batch_instances import SWEBenchInstances


def extract_plan_from_patch(patch: str) -> str:
    ## get plan.md content 
    if "plan.md" not in patch:
        return ""
    lines, in_plan = [], False
    for line in patch.split("\n"):
        if "plan.md" in line and line.startswith("+++"):
            in_plan = True
        elif in_plan:
            if line.startswith("diff "):
                break
            if line.startswith("+") and not line.startswith("+++"):
                lines.append(line[1:])
    return "\n".join(lines)


def load_plans(plan_file: Path) -> tuple[dict[str, str], str, str | None]:
    ## planner results from results, returns (plans, model_short, run_id)
    data = json.loads(plan_file.read_text())
    plans = {r["instance_id"]: r["plan"] for r in data["results"] if r.get("plan")}
    plan_model = data.get("model_short", data.get("model", "unknown"))
    # Extract run id from filename (e.g., results_run2.json -> run2)
    import re
    run_match = re.search(r"_run(\d+)", plan_file.stem)
    plan_run = f"_run{run_match.group(1)}" if run_match else ""
    return plans, plan_model, plan_run


def collect_results(output_dir: Path, extract_plans: bool = False) -> list[dict]:
    ## all results
    results = []
    for instance_dir in output_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        instance_id = instance_dir.name
        traj_file = instance_dir / f"{instance_id}.traj"
        pred_file = instance_dir / f"{instance_id}.pred"
        
        # Get patch from .pred file (JSON with model_patch field)
        patch = ""
        if pred_file.exists():
            pred_data = json.loads(pred_file.read_text())
            patch = pred_data.get("model_patch", "")
        
        result = {
            "instance_id": instance_id,
            "trajectory": json.loads(traj_file.read_text()) if traj_file.exists() else None,
            "patch": patch
        }
        if extract_plans:
            result["plan"] = extract_plan_from_patch(result["patch"])
        results.append(result)
    return results


class SWEBenchWithPlans(SWEBenchInstances):
    ## coder plan prompt support
    def __init__(self, plans: dict[str, str] | None = None, shuffle: bool = False, **kwargs):
        super().__init__(shuffle=shuffle, **kwargs)
        self._plans = plans or {}
    
    def __iter__(self):
        for instance in super().__iter__():
            if instance.problem_statement.id in self._plans:
                plan = self._plans[instance.problem_statement.id]
                instance.problem_statement.text += f"\n\n<execution_plan>\n{plan}\n</execution_plan>"
            yield instance


def run_inference(mode: str, model: str, islice: str = "0:1", subset: str = "lite",
                  config: str | None = None, run_id: str | None = None,
                  plan_file: str | None = None, num_workers: int = 1,
                  shuffle: bool = False) -> list[dict]:
    ## run inference (codebase agentic)
    model, model_short = resolve_model(model)
    model_dir = model.replace("/", "_")
    config_path = resolve_path(config) if config else DEFAULT_CONFIGS[mode]
    
    # plans for coder mode
    plans, plan_model, plan_run = {}, None, ""
    if mode == "coder" and plan_file:
        plans, plan_model, plan_run = load_plans(resolve_path(plan_file))
        print(f"Loaded {len(plans)} plans from {plan_model}{plan_run}")
    
    # dirs
    run_suffix = f"_run{run_id}" if run_id else ""
    if mode == "planner":
        traj_dir = TRAJECTORIES_DIR / f"{model_dir}{run_suffix}"
        results_dir = RESULTS_DIR / model_dir
    else:
        out_name = f"{model_short}_with_{plan_model}_plan{plan_run}" if plan_model else model_short
        traj_dir = CODER_TRAJECTORIES_DIR / f"{out_name}{run_suffix}"
        results_dir = CODER_RESULTS_DIR / out_name
    
    traj_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # load/configure agent
    config_data = yaml.safe_load(config_path.read_text())
    config_data["agent"]["model"]["name"] = model
    agent_config = DefaultAgentConfig.model_validate(config_data["agent"])
    
    # inputs
    if mode == "coder" and plans:
        instances = SWEBenchWithPlans(plans=plans, subset=subset, split="test", slice=islice, shuffle=shuffle)
    else:
        instances = SWEBenchInstances(subset=subset, split="test", slice=islice, shuffle=shuffle)
    
    # run
    runner = RunBatch.from_config(RunBatchConfig(
        instances=instances, agent=agent_config, output_dir=traj_dir, num_workers=num_workers))
    runner.main()

    # save
    results = collect_results(traj_dir, extract_plans=(mode == "planner"))
    output = {
        "mode": mode, "model": model, "model_short": model_short,
        "subset": subset, "config_file": str(config_path), "results": results
    }
    if plan_model:
        output["plan_model"] = plan_model
    
    filename = f"results_run{run_id}.json" if run_id else "results.json"
    results_file = results_dir / filename
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nsaved to: {results_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run planner or coder inference")
    parser.add_argument("--mode", default="planner", choices=["planner", "coder"])
    parser.add_argument("--model", default="haiku", help="haiku, sonnet, opus, or full name")
    parser.add_argument("--slice", default="0:1", help="Instance slice, e.g. 0:3")
    parser.add_argument("--subset", default="lite", choices=["lite", "verified", "full"])
    parser.add_argument("--config", help="Config file (default based on mode)")
    parser.add_argument("-k", type=int, default=1, help="Number of runs")
    parser.add_argument("-w", "--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--plan-file", help="Planner results to inject (coder mode)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle instances for diverse sampling")
    args = parser.parse_args()

    for i in range(1, args.k + 1):
        if args.k > 1:
            print(f"\n{'='*50}\nRun {i}/{args.k}\n{'='*50}")
        run_inference(args.mode, args.model, args.slice, args.subset, 
                     args.config, run_id=str(i) if args.k > 1 else None, 
                     plan_file=args.plan_file, num_workers=args.workers,
                     shuffle=args.shuffle)
