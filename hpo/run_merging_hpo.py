import optuna


def create_mergekit_yaml(trial, 
                         layer_length, 
                         layer_overlap, 
                         base_model="cognitivecomputations/dolphin-2.2-70b",
                         max_layers=80):
    
    config_file_name = "configs/MegaDolphin-Optuna-" + str(trial.number) + ".yaml"
    step_size = int(layer_length * (1 - layer_overlap))
    layer_start_points = [i for i in range(0, max_layers-step_size, step_size)]
    layer_end_points = [i + layer_length for i in layer_start_points]
    layer_end_points[-1] = max_layers
    print(layer_length, layer_overlap, step_size, layer_start_points, layer_end_points)

    yaml_config = """dtype: float16\nmerge_method: passthrough\nslices:"""

    for start, end in zip(layer_start_points, layer_end_points):
        yaml_config += f"""\n- sources:\n  - layer_range: [{start}, {end}]\n    model: {base_model}"""

    # Save config as yaml file
    with open(config_file_name, 'w', encoding="utf-8") as f:
        f.write(yaml_config)


def objective(trial):

    # sample hyperparameters
    layer_length = trial.suggest_int('layer_length', 1, 30)
    layer_overlap = trial.suggest_float('layer_overlap', 0.0, 0.5)

    # create mergekit config
    create_mergekit_yaml(trial, layer_length, layer_overlap)

    # TODO: create model here, e.g.
    #!mergekit-yaml config.yaml merge --copy-tokenizer --allow-crimes --out-shard-size 1B --lazy-unpickle

    # TODO: evaluate model here
    eval_loss = 1

    return eval_loss


study = optuna.create_study()
study.optimize(objective, n_trials=500)