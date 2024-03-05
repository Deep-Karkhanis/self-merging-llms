# Self Merging LLMs
Efficient Self-Merging for Large Language Models

## Running LLM-Evals 
- Clone lm-evaluation-harness and checkout to commit consistent with the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
  ```
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git 
  cd lm-evaluation-harness
  git checkout b281b0921b636bc36ad05c0b0b0763bd6dd43463
  ```
- Clone our repository and run the docker image. If needed, provide a comma separated list of GPU ids to run docker across multiple GPUs
  ```
  cd ../
  git clone https://github.com/Deep-Karkhanis/self-merging-llms.git
  ./self-merging-llms/scripts/lambdadocker-default.sh 0
  ```
- Install lm-evaluation-harness in docker
  ```
  cd /repos/lm-evaluation-harness
  pip install -e .
  ```
- Use script [llm_leaderboard_eval.py](https://github.com/Deep-Karkhanis/self-merging-llms/blob/main/llm_leaderboard_eval.py) to run evals. Un-comment tasks you want evaluated
  ```
  TASK_TO_LIST = {
    # "arc": "arc_challenge",
    # "hellaswag": "hellaswag",
    # "truthfulqa": "truthfulqa-mc",
    # "winogrande": "winogrande",
    "gsm8k": "gsm8k",
    # "drop": "drop",
  }
  ```
- Run evals. Specify `--batch-size` parameter if GPU usage isn't optimal 
  ```
  cd /repos/self-merging-llms
  python ./llm_leaderboard_eval.py --model-path DeepKarkhanis/NeuralPipe-7B-slerp --output-path ./llm_eval_results/DeepKarkhanis-NeuralPipe-7B-slerp --eval-harness-path /repos/lm-evaluation-harness --batch-size 32
  ```
