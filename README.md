# Self Merging LLMs
Efficient Self-Merging for Large Language Models

## Launching docker to run the repo
- Clone our repository 
  ```
  git clone https://github.com/Deep-Karkhanis/self-merging-llms.git
  cd self-merging-llms
  ```
- Run the docker image. If needed, provide a comma separated list of GPU ids to run docker across multiple GPUs
  ```
  ./scripts/lambdadocker-default.sh 0
  ```
- The lm-evaluation-harness directory is a copy of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness.git) repo checked out to the commit `b281b0921b636bc36ad05c0b0b0763bd6dd43463`, consistent with the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- Install lm-evaluation-harness in docker
  ```
  cd /repos/self-merging-llms/lm-evaluation-harness
  pip install -e .
  cd ../
  ```
  
## Running HPO on ARC Val set
  ```
  python main.py
  ```

## Running LLM-Evals 
- Use script [llm_leaderboard_eval.py](https://github.com/Deep-Karkhanis/self-merging-llms/blob/main/llm_leaderboard_eval.py) to run evals. Un-comment tasks you want evaluated
  ```
  TASK_TO_LIST = {
    "arc": "arc_challenge",
    # "hellaswag": "hellaswag",
    # "truthfulqa": "truthfulqa-mc",
    # "winogrande": "winogrande",
    # "gsm8k": "gsm8k",
    # "drop": "drop",
  }
  ```
- Run evals. Specify `--batch-size` parameter if GPU usage isn't optimal 
  ```
  cd /repos/self-merging-llms
  python ./llm_leaderboard_eval.py --model-path DeepKarkhanis/NeuralPipe-7B-slerp --output-path ./llm_eval_results/DeepKarkhanis-NeuralPipe-7B-slerp --eval-harness-path /repos/lm-evaluation-harness --batch-size 32
  ```
