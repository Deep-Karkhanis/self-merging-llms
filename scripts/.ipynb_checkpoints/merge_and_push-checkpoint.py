import logging
from argparse import ArgumentParser
import huggingface_hub as hub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft.config import PeftConfig
from peft.peft_model import PeftModel
from internal.utils import init_logging


def push_model(base_model_path: str, destination: str,
               tokenizer_path: str = None, peft_path: str = None):
    tokenizer_path = tokenizer_path or base_model_path
    logging.info(f'Loading tokenizer from: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logging.info(f'Pushing tokenizer to {destination}')
    if destination.startswith('abacusai/'):
        tokenizer.push_to_hub(destination, safe_serialization=True)
    else:
        tokenizer.save_pretrained(destination, safe_serialization=True)
    logging.info('Pushed tokenizer')
    torch.set_default_device('cpu')
    logging.info(f'Loading model from {base_model_path}')
    final_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
    if peft_path:
        logging.info(f'Loading Peft adapter from {peft_path}')
        peft_config = PeftConfig.from_pretrained(peft_path)
        peft_config.init_lora_weights = False
        final_model = PeftModel.from_pretrained(final_model, peft_path, config=peft_config)
        logging.info('Merging and unloading peft adapter')
        final_model = final_model.merge_and_unload()
    logging.info(f'{type(final_model)} with config:\n{final_model.config.to_json_string()}')
    logging.info(f'Starting push to {destination}')
    if destination.startswith('abacusai/'):
        final_model.push_to_hub(destination, safe_serialization=True)
    else:
        final_model.save_pretrained(destination, safe_serialization=True)
    logging.info('Done')


def main():
    init_logging()

    parser = ArgumentParser(description='Script to push a model to HF, optionally merging PEFT weights.')
    parser.add_argument('--base', required=True, help='Base model path')
    parser.add_argument('--tokenizer', help='Tokenizer path if it is not present in base model')
    parser.add_argument('--peft', help='PEFT adapter model that will be merged before pushing')
    parser.add_argument('--destination', required=True, help='HF destination')
    parser.add_argument('--token', help='HF login token, if explicit login is required.')
    args = parser.parse_args()

    if args.token:
        hub.login(args.token, add_to_git_credential=False)

    push_model(args.base, args.destination, args.tokenizer, args.peft)


if __name__ == '__main__':
    main()
