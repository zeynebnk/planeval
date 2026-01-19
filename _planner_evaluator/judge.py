#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import anthropic

from utils import resolve_model, resolve_path, JUDGE_PROMPT

ISSUE_RE = re.compile(r"<issue>(.*?)</issue>", re.DOTALL)


def extract_problem_statement(result: dict) -> str:
    ## get problem
    history = result.get("trajectory", {}).get("history", [])
    for msg in history:
        if msg.get("message_type") == "observation":
            content = msg.get("content", [])
            if isinstance(content, list) and content:
                if match := ISSUE_RE.search(content[0].get("text", "")):
                    return match.group(1).strip()
    return ""


def judge_plan(client: anthropic.Anthropic, problem_statement: str, 
               golden_plan: str, model_plan: str, model: str) -> dict:
    response = client.messages.create(
        model=model, max_tokens=1024,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            problem_statement=problem_statement,
            golden_plan=golden_plan,
            model_plan=model_plan)}])
    try:
        return json.loads(response.content[0].text.strip())
    except json.JSONDecodeError:
        return {"error": "failed to parse", "raw": response.content[0].text}


def avg(scores: list) -> float:
    return sum(scores) / len(scores) if scores else 0

def run_judge(golden_path: str, model_path: str, judge_model: str = "claude-3-5-haiku-20241022",
              n_runs: int = 1, output_path: str | None = None) -> dict:
    ## run judging (file or dir w results)
    client = anthropic.Anthropic()
    golden = json.loads(resolve_path(golden_path).read_text())
    golden_plans = {r["instance_id"]: r for r in golden["results"]}
    model_path = resolve_path(model_path)
    
    results_files = sorted(model_path.glob("results*.json")) if model_path.is_dir() else [model_path]
    if not results_files:
        print(f"no results*.json files found in {model_path}")
        return {}
    
    if model_path.is_dir():
        print(f"Found {len(results_files)} results files")
    
    all_runs = []
    for results_file in results_files:
        model_results = json.loads(results_file.read_text())
        plan_name = results_file.stem
        
        for judge_idx in range(1, n_runs + 1):
            if len(results_files) > 1 or n_runs > 1:
                print(f"\n{'='*50}\nplan: {plan_name} | judge: {judge_idx}/{n_runs}\n{'='*50}")
            
            judgments = []
            for model_result in model_results["results"]:
                instance_id = model_result["instance_id"]
                if instance_id not in golden_plans:
                    continue
                
                print(f"judging {instance_id}...")
                judgment = judge_plan(
                    client, extract_problem_statement(model_result),
                    golden_plans[instance_id]["plan"], model_result["plan"], judge_model)
                
                judgments.append({"instance_id": instance_id, "judgment": judgment})
                print(f"  overall: {judgment.get('overall', 'N/A')}")
            
            all_runs.append({
                "plan_run": plan_name, "judge_run": judge_idx,
                "avg_score": avg([j["judgment"].get("overall", 0) for j in judgments]),
                "judgments": judgments})
    
    # Summary
    print(f"\n{'='*60}\nSUMMARY: {len(results_files)} plans × {n_runs} judges\n{'='*60}")
    
    by_plan = defaultdict(list)
    for run in all_runs:
        by_plan[run["plan_run"]].append(run["avg_score"])
    
    print(f"\n{'plan':<20} | " + " | ".join(f"J{i}" for i in range(1, n_runs + 1)) + " | Mean")
    print("-" * (30 + n_runs * 7))
    
    all_scores = []
    for plan, scores in sorted(by_plan.items()):
        all_scores.extend(scores)
        print(f"{plan:<20} | " + " | ".join(f"{s:.2f}" for s in scores) + f" | {avg(scores):.2f}")
    
    grand_mean = avg(all_scores)
    print(f"\ngrand mean: {grand_mean:.2f}/5")
    
    # Save
    output = {
        "judge_model": judge_model, "golden_path": str(golden_path), "model_path": str(model_path),
        "n_plans": len(results_files), "n_judges": n_runs, "grand_mean": grand_mean, "runs": all_runs}
    
    out_file = resolve_path(output_path) if output_path else (
        model_path / "all_judgments.json" if model_path.is_dir() 
        else model_path.parent / f"{model_path.stem}_judgments.json")
    
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nsaved to: {out_file}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Judge for plan comparison")
    parser.add_argument("--golden", required=True, help="Path to golden results JSON")
    parser.add_argument("--model", required=True, help="Path to results file or directory")
    parser.add_argument("--judge", default="haiku", help="haiku, sonnet, opus, or full name")
    parser.add_argument("-n", type=int, default=1, help="Judge runs per results file")
    parser.add_argument("--output", help="Output path")
    args = parser.parse_args()
    
    judge_model, _ = resolve_model(args.judge)
    run_judge(args.golden, args.model, judge_model, args.n, args.output)
