import optuna


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


def objective(trial):

    # sample hyperparameters
    layer_length = trial.suggest_int('layer_length', 1, 30)
    layer_overlap = trial.suggest_float('layer_overlap', 0.0, 0.5)

    # create mergekit config
    config_file_name = "configs/MegaDolphin-Optuna-" + str(trial.number) + ".yaml"
    create_mergekit_yaml(config_file_name, 
                         layer_length, 
                         layer_overlap,
                         base_model="cognitivecomputations/dolphin-2.2-70b",
                         max_layers=80)

    try:
        # TODO: run mergekit command to create model here, given the config config_file_name, e.g. equivalent to
        #!mergekit-yaml $config_file_name merge --copy-tokenizer --allow-crimes --out-shard-size 1B --lazy-unpickle

        # TODO: evaluate model here, and replace 0.5 with the true eval loss
        eval_loss = 0.5

    except Exception as e:
        # if we fail to create and evaluate the model, then print the error and return the worst possible loss
        print(e)
        return 1.0

    # return loss on the eval set
    return eval_loss


study = optuna.create_study()
study.optimize(objective, n_trials=50)