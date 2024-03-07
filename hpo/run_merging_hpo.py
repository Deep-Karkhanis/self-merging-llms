import optuna
import json
import logging
import yaml
import os

from llm_leaderboard_eval import run_eval_suite
from argparse import Namespace
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, add_merge_options


def create_mergekit_yaml(config_file_name, 
                         layer_length, 
                         layer_overlap, 
                         base_model,
                         max_layers):

    step_size = int(layer_length * (1 - layer_overlap))
    if step_size == 0:
        step_size = 1
    layer_start_points = [i for i in range(0, max_layers-step_size, step_size)]
    layer_end_points = [i + layer_length for i in layer_start_points]
    layer_end_points[-1] = max_layers

    yaml_config = """dtype: float16\nmerge_method: passthrough\nslices:"""

    for start, end in zip(layer_start_points, layer_end_points):
        yaml_config += f"""\n- sources:\n  - layer_range: [{start}, {end}]\n    model: {base_model}"""

    # Save config as yaml file
    with open(config_file_name, 'w', encoding="utf-8") as f:
        f.write(yaml_config)


def run_mergekit_yaml(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
    verbose: bool):

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    with open(config_file, "r", encoding="utf-8") as file:
        config_source = file.read()

    merge_config: MergeConfiguration = MergeConfiguration.model_validate(
        yaml.safe_load(config_source)
    )

    run_merge(
        merge_config,
        out_path,
        options=merge_options,
        config_source=config_source)


def objective(trial):

    # sample hyperparameters
    layer_length = trial.suggest_int('layer_length', 1, 30)
    layer_overlap = trial.suggest_float('layer_overlap', 0.0, 0.5)

    # create config and output folders
    config_file_name = "configs/MegaDolphin-Optuna-" + str(trial.number) + ".yaml"
    output_folder_name = "models/MegaDolphin-Optuna-" + str(trial.number)

    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    create_mergekit_yaml(config_file_name, 
                         layer_length, 
                         layer_overlap,
                         base_model="cognitivecomputations/dolphin-2.2-70b",
                         max_layers=80)

    try:
        # TODO: run mergekit command to create model here, given the config config_file_name, e.g. equivalent to

        merge_options = MergeOptions()
        run_mergekit_yaml(
            config_file=config_file_name,
            out_path=output_folder_name,
            merge_options=merge_options,
            verbose=True)

        # call run_eval_suite(args) with the following args:
        # python ./llm_leaderboard_eval.py  --model-path DeepKarkhanis/NeuralPipe-7B-slerp 
        #                                   --output-path ./llm_eval_results/DeepKarkhanis-NeuralPipe-7B-slerp 
        #                                   --eval-harness-path /repos/lm-evaluation-harness --batch-size 32
        llm_eval_args = Namespace()
        llm_eval_args.model_path = "DeepKarkhanis/NeuralPipe-7B-slerp" # HF location of the model
        llm_eval_args.peft_path = None
        llm_eval_args.tokenizer_path = None # tokenizer at model_path
        llm_eval_args.output_path = f"./llm_eval_results/DeepKarkhanis-NeuralPipe-7B-slerp/trial-{str(trial.number)}"
        llm_eval_args.eval_harness_path = "./lm-evaluation-harness"
        llm_eval_args.device = None
        llm_eval_args.batch_size = 32
        llm_eval_args.bench_on_val = True
        llm_eval_args.force_rerun = True

        eval_log = run_eval_suite(llm_eval_args)
        # read llm_eval_args.output_path/arc as a json file
        val_score = json.load(open(f"{llm_eval_args.output_path}/arc", "r"))['results']['arc_challenge']['acc_norm']
        eval_loss = 1.0 - val_score
        # eval_loss = 0.5
    
    except Exception as e:
        # if we fail to create and evaluate the model, then print the error and return the worst possible loss
        print(e)
        return 1.0

    return eval_loss


def run_merging_hpo():
    if not os.path.exists("configs"):
        os.makedirs("configs")

    study = optuna.create_study()
    study.optimize(objective, n_trials=50)