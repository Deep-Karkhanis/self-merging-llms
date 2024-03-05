# Sample usage: python reasoning/eval/llm_leaderboard_eval.py --model-path abacusai/Giraffe-beta-13b-32k --output-path ../llm_eval_suite/Giraffe-beta-13b-32k

import argparse
import os
import json

TASK_TO_LIST = {
    "arc": "arc_challenge",
    # "hellaswag": "hellaswag",
    # "truthfulqa": "truthfulqa_mc",
    # "mmlu": "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions",
    # "winogrande": "winogrande",
    # "gsm8k": "gsm8k",
    # "drop": "drop",
}
TASK_TO_NUM_SHOTS = {
    "arc": 25,
    "hellaswag": 10,
    "truthfulqa": 0,
    "mmlu": 5,
    "winogrande": 5,
    "gsm8k": 5,
    "drop": 3,
}

def run_eval_suite(args):

    model_args = {
    'pretrained': args.model_path,
    'use_accelerate': True
    }
    if args.tokenizer_path is not None:
        model_args['tokenizer'] = args.tokenizer_path
    if args.peft_path is not None:
        model_args['peft'] = args.peft_path
    model_args_str = ','.join(f'{k}={v}' for k, v in model_args.items())

    os.system(f"mkdir -p {args.output_path}")

    eval_log = {}
    # Check if file "{args.output_path}/eval_log.json" exists
    # If it does, then we should only run the tasks that we haven't run yet
    # If it doesn't, then we should run all of the tasks
    if os.path.isfile(f"{args.output_path}/eval_log.json"):
        eval_log = json.load(open(f"{args.output_path}/eval_log.json", "r"))

    for task in TASK_TO_LIST.keys():
        if task in eval_log:
            print(f"Skipping {task} because it has already been run")
            continue
        print(f"Running {task}...\n")
        num_shots = TASK_TO_NUM_SHOTS[task]
        task_list = TASK_TO_LIST[task]
        # Create file args.output_path/task if it doesn't exist
        open(f"{args.output_path}/{task}", "w").write('')

        # command = f'{args.eval_harness_path} --model=hf --model_args="{model_args_str}" --tasks={task_list} --num_fewshot={num_shots} --batch_size=2 --output_path={args.output_path}/{task}'
        command = f'python {args.eval_harness_path}/main.py --model=hf-causal-experimental --model_args="{model_args_str}" --tasks={task_list} --num_fewshot={num_shots} --batch_size={args.batch_size} --output_path={args.output_path}/{task}'

        if args.device is not None:
            command += f" --device={args.device}"
        print(command + '\n')
        ret = os.system(command)

        if ret != 0:
            print(f"ERROR: {task} failed to run")
        else:
            eval_log[task] = True
            json.dump(eval_log, open(f"{args.output_path}/eval_log.json", "w"), indent=4)
    return

# Create a script that takes a few arguments and runs the evaluation suite
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on a leaderboard')
    parser.add_argument('--model-path', type=str, help='Path to the base model')
    parser.add_argument('--peft-path', type=str, default=None, help='If applicable, path to the peft model')
    parser.add_argument('--tokenizer-path', type=str, default=None, help='If applicable, path to the tokenizer')
    parser.add_argument('--output-path', type=str, help='Path to the output directory')
    parser.add_argument('--eval-harness-path', type=str, default='~/lm-evaluation-harness', help='Path to the eval_harness directory')
    parser.add_argument('--device', type=str, default=None, help='Device to run on eg cuda:0,1')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size to use for evaluation')

    print("\n\n=============================================================")
    print("INFO: Make sure you have the lm-evaluation-harness repo cloned in your home directory and installed with \"pip install -e .\" ")
    print("=============================================================\n\n")
    args = parser.parse_args()

    run_eval_suite(args)
