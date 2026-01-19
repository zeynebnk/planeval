from pathlib import Path

# dirs
ROOT = Path(__file__).parent.resolve()
SWEAGENT_ROOT = ROOT.parent

RESULTS_DIR = ROOT / "results"
CODER_RESULTS_DIR = ROOT / "coder_results"
TRAJECTORIES_DIR = SWEAGENT_ROOT / "trajectories"
CODER_TRAJECTORIES_DIR = SWEAGENT_ROOT / "trajectories_coder"

# models
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",  
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}
MODEL_SHORT = {v: k for k, v in MODEL_ALIASES.items()}

# configs
DEFAULT_CONFIGS = {
    "planner": SWEAGENT_ROOT / "config/custom/planner_config.yaml",
    "coder": SWEAGENT_ROOT / "config/custom/coder.yaml",
}


def resolve_model(model: str) -> tuple[str, str]:
    full = MODEL_ALIASES.get(model, model)
    short = MODEL_SHORT.get(full, full)
    return full, short


def resolve_path(path: str | Path, relative_to: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else relative_to / p


JUDGE_PROMPT = """You are an expert evaluator comparing implementation plans for software engineering tasks.

## Task Description
{problem_statement}

## Golden Reference Plan (Ground Truth)
{golden_plan}

## Model-Generated Plan (To Evaluate)
{model_plan}

Your task is to evaluate the quality of the model generated plan against the golden reference plan.

## Evaluation Criteria
Score the model plan on each criterion from 1-5 (1=poor, 5=excellent):

1. **Problem Understanding**: Does it correctly identify the root cause and problem?
2. **Coverage**: Does the model plan cover each item and component in the golden reference plan? Does it cover all necessary changes?
3. **Detail**: Does the model plan provide sufficient detail and clarity, to the level of the golden reference?
4. **Solution Correctness**: Is the proposed solution technically correct and aligned with the golden reference?
5. **Code Localization**: Does it identify the correct files and functions to modify, compared to the golden reference?
6. **Usefulness**: Is the plan useful and actionable? 
7. **Overall**: The overall score is the average of the above criteria.

## Output Format
Respond with ONLY a JSON object:
{{"problem_understanding": <1-5>, "coverage": <1-5>, "detail": <1-5>, "solution_correctness": <1-5>, "code_localization": <1-5>, "usefullness": <1-5>,"overall": <1-5>, "reasoning": "<brief explanation>"}}
"""
