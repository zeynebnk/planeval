
## planner inference 
python run_inference.py --model haiku --slice 0:3 -k 3

## golden inference (special prompt/config) 
python run_inference.py --model opus --slice 0:3 --config ../config/custom/golden_config.yaml

## coder inference (baseline)
python run_inference.py --mode coder --model haiku --slice 0:3

## coder inference (+ plan)
python run_inference.py --mode coder --model haiku --slice 0:3 \
    --plan-file results/claude-opus-4-20250514/results.json

# judging
python judge.py --golden results/claude-opus-4-20250514/results.json \
    --model results/claude-haiku-4-5-20251001/ --judge haiku -n 3

## coder eval 
python run_eval.py coder_results/haiku/preds.json
